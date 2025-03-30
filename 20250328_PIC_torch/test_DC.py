#%% test directional coupler

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DC_CMT import DirectionalCouplerCoupledMode, SegmentedDirectionalCoupler


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
optimizer = torch.optim.Adam([dc.kappa_scale, dc.delta_scale], lr=1e-3)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass through the directional coupler
    output_field = dc(input_field)
    
    # Compose complex fields for each waveguide
    output_complex_wg1 = torch.complex(output_field[0, 0, :], output_field[0, 1, :])
    output_complex_wg2 = torch.complex(output_field[0, 2, :], output_field[0, 3, :])
    
    # Compute power (magnitude squared)
    # Compute power (magnitude squared)
    power_wg1 = torch.abs(output_complex_wg1)**2
    power_wg2 = torch.abs(output_complex_wg2)**2

    # Compute total power and avoid division by zero
    total_power = power_wg1 + power_wg2 + 1e-8

    # Define loss as the mean squared deviation of the power ratio in WG1 from 45%
    loss = torch.mean((power_wg1 / total_power - 0.45)**2)

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
kappa = dc.kappa_normalized * 2 * torch.pi / dc.wavelength_points
delta_beta = dc.delta_beta_normalized * 2 * torch.pi / dc.wavelength_points
gamma = torch.sqrt(kappa**2 + (delta_beta / 2)**2)

L = dc.L
T11 = torch.cos(gamma * L) + 1j * (delta_beta / (2 * gamma)) * torch.sin(gamma * L)
T12 = -1j * (kappa / gamma) * torch.sin(gamma * L)

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
