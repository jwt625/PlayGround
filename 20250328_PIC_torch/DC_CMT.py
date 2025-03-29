#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class DirectionalCouplerCoupledMode(nn.Module):
    def __init__(self, wavelength_points, L=0.001):
        """
        A directional coupler modeled using coupled mode theory.
        
        wavelength_points: array or list of wavelength points (e.g., in meters)
        L: length of the directional coupler (meters)
        """
        super(DirectionalCouplerCoupledMode, self).__init__()
        # Register wavelength points as a buffer (non-learnable)
        self.register_buffer('wavelength_points', torch.tensor(wavelength_points, dtype=torch.float32))
        self.L = L  # coupler length
        
        # Normalize wavelength points to [0,1] for interpolation
        wl_norm = (wavelength_points - wavelength_points[0]) / (wavelength_points[-1] - wavelength_points[0])
        
        # Learnable parameters: coupling coefficient and detuning for each wavelength
        # Units should be consistent (e.g., 1/m)
        # Define points for quadratic interpolation
        x = np.array([0, 0.5, 1.0])  # normalized positions
        y = np.array([100e-3, 100e-3, 100e-3])  # coefficients at min, center, max
        
        kappa_normalized = self.langrange_polynomial(x, y, wl_norm)
        kappa_normalized = torch.tensor(kappa_normalized, dtype=torch.float32)
        self.kappa_normalized = nn.Parameter(kappa_normalized)      # coupling rate
        # Calculate quadratic detuning using Lagrange interpolation
        # Define points for quadratic interpolation
        x = np.array([0, 0.5, 1.0])  # normalized positions
        y = np.array([2e-3, 4e-3, 8e-3])  # coefficients at min, center, max
        
        # Combine to get detuning coefficients
        detuning_coeff = self.langrange_polynomial(x, y, wl_norm)
        
        delta_normalized = torch.tensor(detuning_coeff, dtype=torch.float32)         # convert to torch tensor
        self.delta_beta_normalized = nn.Parameter(delta_normalized)

    def langrange_polynomial(self, x, y, wl_norm):
        """
        x: input tensor of shape (batch_size, 4, N)
        y: input tensor of shape (batch_size, 4, N)
        wl_norm: input tensor of shape (batch_size, N)
        """
        l0 = (wl_norm - x[1]) * (wl_norm - x[2]) / ((x[0] - x[1]) * (x[0] - x[2]))
        l1 = (wl_norm - x[0]) * (wl_norm - x[2]) / ((x[1] - x[0]) * (x[1] - x[2]))
        l2 = (wl_norm - x[0]) * (wl_norm - x[1]) / ((x[2] - x[0]) * (x[2] - x[1]))
        
        y_interp = y[0]*l0 + y[1]*l1 + y[2]*l2
        
        return y_interp
        
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
           where 4 channels represent [real1, imag1, real2, imag2]
           and N is the number of wavelength points.
        """
        # print("Input shape:", x.shape)
        
        # Convert to complex representation
        x_complex = torch.complex(x[:, 0], x[:, 1])  # waveguide 1
        y_complex = torch.complex(x[:, 2], x[:, 3])  # waveguide 2
        z = torch.stack([x_complex, y_complex], dim=1)  # shape: (batch_size, 2, N)
        # print("Stacked complex shape:", z.shape)
        
        # Compute gamma from coupling coefficient and detuning for each wavelength:
        kappa = self.kappa_normalized * 2 * torch.pi / self.wavelength_points
        delta_beta = self.delta_beta_normalized * 2 * torch.pi / self.wavelength_points
        gamma = torch.sqrt(kappa**2 + (delta_beta / 2)**2)
        # print("Gamma shape:", gamma.shape)
        
        # Calculate matrix elements based on coupled mode theory for each wavelength
        L = self.L
        T11 = torch.cos(gamma * L) + 1j * (delta_beta / (2 * gamma)) * torch.sin(gamma * L)
        T12 = -1j * (kappa / gamma) * torch.sin(gamma * L)
        T21 = T12
        T22 = torch.cos(gamma * L) - 1j * (delta_beta / (2 * gamma)) * torch.sin(gamma * L)
        # print("T11 shape:", T11.shape)
        
        # Form the 2x2 coupling matrix for each wavelength
        coupling_matrix = torch.stack([
            torch.stack([T11, T12]),
            torch.stack([T21, T22])
        ])  # shape: (2, 2, N)
        # print("Coupling matrix shape:", coupling_matrix.shape)
        
        # Expand coupling matrix to operate on each batch:
        # Reshape to (1, 2, 2, N) and then expand to (batch_size, 2, 2, N)
        coupling_matrix = coupling_matrix.unsqueeze(0)
        coupling_matrix = coupling_matrix.expand(x.shape[0], -1, -1, -1)
        # print("Expanded coupling matrix shape:", coupling_matrix.shape)
        
        # Rearrange input for multiplication: (batch_size, N, 2)
        z_reshaped = z.permute(0, 2, 1)
        # print("Reshaped input shape:", z_reshaped.shape)
        
        # Apply the coupling matrix for each wavelength:
        # The einsum performs: out[b, n, i] = sum_j coupling_matrix[b, i, j, n] * z_reshaped[b, n, j]
        z_out = torch.einsum('bijn,bnj->bin', coupling_matrix, z_reshaped)
        # print("Output after einsum shape:", z_out.shape)
        
        # Convert back to real/imaginary representation
        # z_out shape is (batch_size, 2, N)
        out = torch.stack([
            z_out[:, 0].real,  # shape: (batch_size, N)
            z_out[:, 0].imag,  # shape: (batch_size, N)
            z_out[:, 1].real,  # shape: (batch_size, N)
            z_out[:, 1].imag   # shape: (batch_size, N)
        ], dim=1)  # shape: (batch_size, 4, N)
        # print("Final output shape:", out.shape)
        return out

#%% Example usage:

# Define wavelength points (e.g., from 1.5 μm to 1.6 μm)
wavelength_points = np.linspace(1.3e-6, 1.7e-6, 2000)

# Create an instance of the updated directional coupler
dc = DirectionalCouplerCoupledMode(wavelength_points, L=0.00000195)

# Create an example input:
# Assume batch_size=1, 4 channels (real/imag for both waveguides), N wavelength points.
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 4, N, dtype=torch.float32)
input_field[:, 0, :] = 1.0  # set the real part of waveguide 1 to 1

# Forward pass through the directional coupler
output_field = dc(input_field)

print("Input shape:", input_field.shape)
print("Output shape:", output_field.shape)
print("Learnable parameters:")
for name, param in dc.named_parameters():
    print(f"  {name}: {param.shape}")



#%% loss and gradient calculation
# Forward pass through the directional coupler
output_field = dc(input_field)

# Compose complex fields from the output (channels: 0/1 for WG1, 2/3 for WG2)
output_complex_wg1 = torch.complex(output_field[0, 0, :], output_field[0, 1, :])
output_complex_wg2 = torch.complex(output_field[0, 2, :], output_field[0, 3, :])

# Compute power (magnitude squared) for each waveguide at each wavelength point
power_wg1 = torch.abs(output_complex_wg1)**2
power_wg2 = torch.abs(output_complex_wg2)**2

# Define loss as the mean squared deviation from a 50:50 split.
# For a perfect 50:50 split, power_wg1 should equal power_wg2.
loss = torch.mean((power_wg1 - power_wg2)**2)

# Backward pass to compute gradients
loss.backward()

# Print the loss and check gradients for the learnable parameters.
print("Loss (deviation from 50:50):", loss.item())
print("Gradient for kappa:", dc.kappa_normalized.grad)
print("Gradient for delta_beta:", dc.delta_beta_normalized.grad)


#%%
# Set up an optimizer for the kappa and delta_beta parameters
optimizer = torch.optim.Adam([dc.kappa_normalized, dc.delta_beta_normalized], lr=1e-3)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass through the directional coupler
    output_field = dc(input_field)
    
    # Compose complex fields for each waveguide
    output_complex_wg1 = torch.complex(output_field[0, 0, :], output_field[0, 1, :])
    output_complex_wg2 = torch.complex(output_field[0, 2, :], output_field[0, 3, :])
    
    # Compute power (magnitude squared)
    power_wg1 = torch.abs(output_complex_wg1)**2
    power_wg2 = torch.abs(output_complex_wg2)**2
    
    # Define loss as the mean squared deviation from a 50:50 split
    loss = torch.mean((power_wg1 - power_wg2)**2)
    
    # Backpropagate to compute gradients
    loss.backward()
    optimizer.step()
    
    # Optionally print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        print(f"  kappa mean: {dc.kappa_normalized.mean().item():.6f}")
        print(f"  delta_beta mean: {dc.delta_beta_normalized.mean().item():.6f}")



# %% plot

output_field = dc(input_field)
import matplotlib.pyplot as plt

# Create figure with subplots
plt.figure(figsize=(15, 10))

# Plot 1: Power transfer
plt.subplot(2, 2, 1)

input_complex_wg1 = torch.complex(input_field[0, 0, :], input_field[0, 1, :])
output_complex_wg1 = torch.complex(output_field[0, 0, :], output_field[0, 1, :])
output_complex_wg2 = torch.complex(output_field[0, 2, :], output_field[0, 3, :])

plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_wg1)**2).detach().numpy(),
         label='Input WG1', linestyle='--', alpha=0.5)
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_wg1)**2).detach().numpy(),
         label='Output WG1')
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_wg2)**2).detach().numpy(),
         label='Output WG2')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('Power Transfer')
plt.legend()
plt.grid(True)

# Plot 2: Coupling parameters
plt.subplot(2, 2, 2)
plt.plot(wavelength_points * 1e9, dc.kappa_normalized.detach().numpy(), 
         label='κ', linestyle='--')
plt.plot(wavelength_points * 1e9, dc.delta_beta_normalized.detach().numpy(), 
         label='Δβ')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Value (normalized to k0)')
plt.title('Coupling Parameters')
plt.legend()
plt.grid(True)

# Plot 3: Phase of coupling matrix elements
plt.subplot(2, 2, 3)
gamma = torch.sqrt(dc.kappa_normalized**2 + (dc.delta_beta_normalized / 2)**2)
L = dc.L
T11 = torch.cos(gamma * L) + 1j * (dc.delta_beta_normalized / (2 * gamma)) * torch.sin(gamma * L)
T12 = -1j * (dc.kappa_normalized / gamma) * torch.sin(gamma * L)

plt.plot(wavelength_points * 1e9, torch.angle(T11).detach().numpy(), 
         label='Phase(T11)', linestyle='--')
plt.plot(wavelength_points * 1e9, torch.angle(T12).detach().numpy(), 
         label='Phase(T12)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Phase (rad)')
plt.title('Phase of Coupling Matrix Elements')
plt.legend()
plt.grid(True)

# Plot 4: Magnitude of coupling matrix elements
plt.subplot(2, 2, 4)
plt.plot(wavelength_points * 1e9, abs(torch.abs(T11).detach().numpy())**2, 
         label='|T11|^2', linestyle='--')
plt.plot(wavelength_points * 1e9, abs(torch.abs(T12).detach().numpy())**2, 
         label='|T12|^2')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Magnitude squared')
plt.title('Magnitude squared of Coupling Matrix Elements')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print coupling parameters
print("\nCoupling Parameters:")
print(f"Coupler length (L): {dc.L:.4f} m")
print(f"κ range: [{dc.kappa_normalized.min().item():.4f}, {dc.kappa_normalized.max().item():.4f}] 1/m")
print(f"Δβ range: [{dc.delta_beta_normalized.min().item():.4f}, {dc.delta_beta_normalized.max().item():.4f}] 1/m")

# %%
