
#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from RingResonator import RingResonatorFP
#%%
# ---------------------------
# Test Script
# ---------------------------

# Define wavelength points (e.g., from 1.5 μm to 1.6 μm)
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)

# Instantiate the ring resonator model with desired parameters
ring_resonator = RingResonatorFP(wavelength_points, L=100e-6, r=0.99, a=0.99)

# Create an input field:
# For this test, use a unit-amplitude field with zero phase.
# The expected shape is (batch_size, 2, N). Here batch_size = 1.
batch_size = 1
N = len(wavelength_points)
input_field = torch.zeros(batch_size, 2, N, dtype=torch.float32)
input_field[:, 0, :] = 1.0  # real part = 1, imaginary part = 0

# Evaluate the model (use torch.no_grad() since we're just testing the spectrum)
with torch.no_grad():
    output_field = ring_resonator(input_field)

# Compose the output complex field from the two channels:
output_complex = torch.complex(output_field[0, 0], output_field[0, 1])
# Compute power (magnitude squared) for each wavelength point
power = torch.abs(output_complex)**2

# Convert wavelengths to nm for plotting:
wavelength_nm = wavelength_points * 1e9

# Plot the transmission spectrum: power vs. wavelength
plt.figure(figsize=(10, 6))
plt.plot(wavelength_nm, power.detach().numpy(), label='Transmission Power')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (a.u.)')
plt.title('Ring Resonator Transmission Spectrum')
plt.legend()
plt.grid(True)
plt.show()

# %%
