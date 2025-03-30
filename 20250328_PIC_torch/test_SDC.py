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
sdc = SegmentedDirectionalCoupler(wavelength_points, total_length=0.00000195, 
                                  num_segments=1)

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




# %% plot input and output

output_field = sdc(input_field)
import matplotlib.pyplot as plt

# Create figure with subplots

plt.figure(figsize=(15, 10))

input_complex_wg1 = torch.complex(input_field[0, 0, :], input_field[0, 1, :])
output_complex_wg1 = torch.complex(output_field[0, 0, :], output_field[0, 1, :])
output_complex_wg2 = torch.complex(output_field[0, 2, :], output_field[0, 3, :])
lw = 3
plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_wg1)**2).detach().numpy(),
         label='Input WG1', linestyle='--', alpha=0.5, linewidth=lw)
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_wg1)**2).detach().numpy(),
         label='Output WG1', linewidth=lw)
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_wg2)**2).detach().numpy(),
         label='Output WG2', linewidth=lw)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('Power Transfer')
plt.legend()
plt.grid(True)



# %% training
# Set up an optimizer for the kappa and delta_beta parameters from all segments
optimizer = torch.optim.Adam(
    [param for segment in sdc.segments for param in [segment.kappa_scale, segment.delta_scale, segment.kappa_shift, segment.delta_shift]], 
    lr=1e-3
)
num_epochs = 1000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass through the directional coupler
    output_field = sdc(input_field)
    
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
    loss = torch.mean((power_wg1 / total_power - 0.50)**2)

    # Backpropagate to compute gradients
    loss.backward()
    optimizer.step()
    
    # Optionally print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        for i, segment in enumerate(sdc.segments):
            print(f"  Segment {i+1} kappa mean: {segment.kappa_scale.mean().item():.6f}")
            print(f"  Segment {i+1} delta_beta mean: {segment.delta_scale.mean().item():.6f}")
            print(f"  Segment {i+1} kappa shift: {segment.kappa_shift.item():.6f}")
            print(f"  Segment {i+1} delta_beta shift: {segment.delta_shift.item():.6f}")

# %% plot the final kappa and delta scale and shift vs segments

plt.figure(figsize=(10, 5))
kappa_scale_values = []
delta_scale_values = []
kappa_shift_values = []
delta_shift_values = []

for segment in sdc.segments:
    kappa_scale_values.append(segment.kappa_scale.detach().numpy())
    delta_scale_values.append(segment.delta_scale.detach().numpy())
    kappa_shift_values.append(segment.kappa_shift.detach().numpy())
    delta_shift_values.append(segment.delta_shift.detach().numpy())

# Convert lists to numpy arrays for plotting
kappa_scale_values = np.array(kappa_scale_values)
delta_scale_values = np.array(delta_scale_values)
kappa_shift_values = np.array(kappa_shift_values)
delta_shift_values = np.array(delta_shift_values)

# Plotting the curves
plt.plot(kappa_scale_values, label='Kappa Scale', marker='o')
plt.plot(delta_scale_values, label='Delta Scale', marker='o')
plt.plot(kappa_shift_values, label='Kappa Shift', marker='o')
plt.plot(delta_shift_values, label='Delta Shift', marker='o')

plt.legend()
plt.xlabel('Segment')
plt.ylabel('Value')
plt.title('kappa and delta scale and shift vs segments')
plt.grid(True)


# %% plot the final kappa and delta beta


kappa_normalized = (sdc.segments[0].kappa_normalized * sdc.segments[0].kappa_scale + sdc.segments[0].kappa_shift) 
delta_beta_normalized = (sdc.segments[0].delta_beta_normalized * sdc.segments[0].delta_scale + sdc.segments[0].delta_shift) 

plt.plot(wavelength_points * 1e9, kappa_normalized.detach().numpy(), 
         label='κ', linestyle='--')
plt.plot(wavelength_points * 1e9, delta_beta_normalized.detach().numpy(), 
         label='Δβ')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Value (normalized to k0)')
plt.title('Coupling Parameters')
plt.legend()
plt.grid(True)

# %%
