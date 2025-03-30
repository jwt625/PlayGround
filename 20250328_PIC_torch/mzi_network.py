#%%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from DC_CMT import DirectionalCouplerCoupledMode
from RingResonator import RingResonatorFP


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
        Delay line layer that applies a phase delay to a single waveguide.
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
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 2, N)
           where 2 channels represent [real, imag] of the waveguide.
        """
        # Convert to complex representation
        x_complex = torch.complex(x[:, 0], x[:, 1])  # single waveguide
        
        # Calculate phase difference using learnable parameters
        k = 2 * torch.pi * self.n_eff / self.wavelength_points
        phase_diff = k * self.delta_L
        
        # Apply phase difference to the waveguide
        x_complex = x_complex * torch.exp(1j * phase_diff)
        
        # Convert back to real/imag representation
        out = torch.stack([
            x_complex.real,    # real part
            x_complex.imag     # imag part
        ], dim=1)  # shape: (batch_size, 2, N)
        
        return out



class MZINetwork(nn.Module):
    def __init__(self, wavelength_points, 
                 delta_L=1e-6, n_eff=1.5,
                 ring_FSR=0.01, ring_Q_total=1e3, ring_Q_ext=2e3, phase_shift=0.0):
        """
        Complete MZI network with learnable parameters
        """
        super(MZINetwork, self).__init__()
        # Compute the round-trip length L. (With neff lumped in, we take L = 1/FSR)
        lambda0 = (wavelength_points[0] + wavelength_points[-1]) / 2
        L = lambda0 / ring_FSR
        r = 1 - (torch.pi * L) / (lambda0 * ring_Q_ext)
        a = np.exp(-ring_FSR / (2 * ring_Q_total))
        if ring_Q_ext > ring_Q_total:
            ring_Q_int = (ring_Q_total * ring_Q_ext) / (ring_Q_ext - ring_Q_total)
            a = 1 - (torch.pi * L) / (lambda0 * ring_Q_int)
        else:
            a = 1.0

        self.dc1 = DirectionalCouplerCoupledMode(wavelength_points, L=0.00000195)
        self.ring = RingResonatorFP(wavelength_points, L=L*1e6, r=r, a=a, phase_shift=phase_shift)
        self.delay = DelayLine(wavelength_points, delta_L, n_eff)
        self.dc2 = DirectionalCouplerCoupledMode(wavelength_points, L=0.00000195)
        self.register_buffer('wavelength_points', torch.tensor(wavelength_points, dtype=torch.float32))
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
        """
        # Split the input tensor
        x = self.dc1(x)

        x1 = x[:, :2, :]  # First two channels for the first path (ring)
        x2 = x[:, 2:, :]  # Last two channels for the second path (delay line)

        # Pass the first part through the ring resonator, and second part through the delay line
        x1 = self.ring(x1)  # Pass through the ring resonator
        x2 = self.delay(x2)

        # Combine the outputs
        # Here, you may need to concatenate or stack the outputs appropriately
        # For example, if the ring output is also of shape (batch_size, 2, N):
        x_combined = torch.cat((x1, x2), dim=1)  # Combine along the channel dimension

        x_combined = self.dc2(x_combined)    # Second directional coupler

        return x_combined



# Example usage
#%%
# Define wavelength points
wavelength_points = np.linspace(1.5e-6, 1.6e-6, 5000)  # 1000 points from 1.5 to 1.6 μm

# Create network
mzi = MZINetwork(wavelength_points, delta_L=100e-6,
                 ring_FSR=0.0105, ring_Q_total=2e2,
                 ring_Q_ext=1e3, phase_shift=np.pi/2)
# mzi.ring.r = torch.nn.Parameter(torch.tensor(4.0))
mzi.ring.a = torch.nn.Parameter(torch.tensor(-3.0))
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

plt.figure(figsize=(10, 12))
plt.subplot(3, 1, 1)
# plt.semilogy(wavelength_points * 1e9, (torch.abs(input_complex_1).detach().numpy())**2, label='Input WG1')
# # plt.semilogy(wavelength_points * 1e9, (torch.abs(input_complex_2).detach().numpy())**2, label='Input WG2')
# plt.semilogy(wavelength_points * 1e9, (torch.abs(output_complex_1).detach().numpy())**2, label='Output WG1')
# plt.semilogy(wavelength_points * 1e9, (torch.abs(output_complex_2).detach().numpy())**2, label='Output WG2')
plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_1).detach().numpy())**2, label='Input WG1')
# plt.plot(wavelength_points * 1e9, (torch.abs(input_complex_2).detach().numpy())**2, label='Input WG2')
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_1).detach().numpy())**2, label='Output WG1')
plt.plot(wavelength_points * 1e9, (torch.abs(output_complex_2).detach().numpy())**2, label='Output WG2')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('MZI Response')
plt.legend()
plt.grid(True)
# Compute the derivative of the power in output waveguide 1
d_power_output_1 = torch.diff(torch.abs(output_complex_1)**2)

# Create a new subplot for the derivative
plt.subplot(3, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(wavelength_points[1:] * 1e9, d_power_output_1.detach().numpy(), label='Derivative of Output WG1', color='orange')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Derivative of Power')
plt.title('Derivative of Power in Output Waveguide 1')
plt.grid(True)
plt.legend()


# check ring resonator response

output_ring = mzi.ring(input_field[:, :2, :])
# plot the output of the ring resonator
# Compose the output complex field from the two channels:
output_complex = torch.complex(output_ring[0, 0], output_ring[0, 1])
# Compute power (magnitude squared) for each wavelength point
power = torch.abs(output_complex)**2
plt.subplot(3, 1, 3)
plt.plot(wavelength_points * 1e9, power.detach().numpy(), label='Ring Resonator Response')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power')
plt.title('Ring Resonator Response')
plt.legend()
plt.grid(True)

plt.show()
# Assuming 'ring' is your instance of RingResonator
for name, param in mzi.ring.named_parameters():
    print(f"{name}: {param.data.numpy()}")


#%% training for better linearity
# -----------------------------
# Define the cost function:
def linearity_loss(transmission):
    """
    transmission: tensor of shape (N,) representing the power vs. wavelength.
    wavelengths: tensor of shape (N,) (assumed to be sorted and equally spaced).
    
    The loss is computed as the mean squared deviation of the derivative from its average,
    over the segments where the derivative is positive.
    """
    # Compute finite difference derivative
    # Use central differences (or simple forward differences)
    dT = (transmission[1:] - transmission[:-1])
    # Assume wavelengths are equally spaced
    dx = 1.0 / transmission.shape[0]  # Assuming x range is from 0 to 1
    dT = dT / dx  # approximate derivative
    
    # Create a mask for points where the slope is greater than 2/3 of the maximum slope
    pos_mask = (dT > (2/3) * dT.max()).float()  # shape: (N-1,)
    
    count = pos_mask.sum()
    # To avoid division by zero if no positive slopes:
    if count.item() < 1:
        return torch.tensor(0.0, requires_grad=True)
    
    # Compute the average derivative over positive-slope points
    avg_pos = (dT * pos_mask).sum() / count
    
    # Compute mean squared deviation over those points
    loss = torch.sum(((dT - avg_pos) ** 2)/ (avg_pos**2) * pos_mask) / count
    return loss

#%% Adam optimizer
import torch.optim as optim

# Set up optimizer to train the parameters of the ring resonator (or entire MZI)
optimizer = optim.Adam(mzi.ring.parameters(), lr=1e-1)
num_epochs = 1000

# Training loop to minimize the linearity loss (improving linearity of positive slope regions)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Get the transmission spectrum (power vs. wavelength)
    output_field = mzi(input_field)  # shape: (batch_size, N)
    # For simplicity, take the first sample in the batch:
    # Compose the output complex field from the two channels:
    output_complex = torch.complex(output_field[0, 0], output_field[0, 1])
    # Compute power (magnitude squared) for each wavelength point
    power = torch.abs(output_complex)**2
    
    # Compute the cost function (linearity loss)
    loss = linearity_loss(power)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6e}")
        # Assuming 'ring' is your instance of RingResonator
        for name, param in mzi.ring.named_parameters():
            print(f"{name}: {param.data.numpy()}")



#%%
# After training, plot the transmission spectrum and its derivative.
with torch.no_grad():
    output_field = mzi(input_field)  # shape: (batch_size, N)
    # For simplicity, take the first sample in the batch:
    # Compose the output complex field from the two channels:
    output_complex = torch.complex(output_field[0, 0], output_field[0, 1])
    # Compute power (magnitude squared) for each wavelength point
    power = torch.abs(output_complex)**2
    power_np = power.detach().numpy()
    wl_np = mzi.wavelength_points.detach().numpy()
    dx = wl_np[1] - wl_np[0]
    dpower = np.diff(power_np) / dx
    
plt.figure(figsize=(12, 10))
plt.subplot(3,1,1)
plt.plot(wl_np*1e9, power_np, label='Transmission Power')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (a.u.)')
plt.title('MZI Transmission Spectrum')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(wl_np[:-1]*1e9, dpower, label='Derivative', color='orange')
plt.xlabel('Wavelength (nm)')
plt.ylabel('dPower/dλ')
plt.title('Derivative of the Transmission Spectrum')
plt.legend()
plt.grid(True)

# Compute the ring response
ring_response = mzi.ring(input_field)  # Pass the first part through the ring resonator
ring_response_complex = torch.complex(ring_response[0, 0], ring_response[0, 1])
ring_power = torch.abs(ring_response_complex)**2
ring_power_np = ring_power.detach().numpy()

plt.subplot(3, 1, 3)
plt.plot(wl_np * 1e9, ring_power_np, label='Ring Response', color='green')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Power (a.u.)')
plt.title('Ring Resonator Response')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()


# Assuming 'ring' is your instance of RingResonator
for name, param in mzi.ring.named_parameters():
    print(f"{name}: {param.data.numpy()}")

#%% trying LBFGS
# Assuming your model is called "mzi" and your loss function is "linearity_loss"

# Use LBFGS optimizer
optimizer = torch.optim.LBFGS(mzi.ring.parameters(), lr=1e-2, max_iter=20, history_size=10)

num_epochs = 200

def closure():
    optimizer.zero_grad()
    # Get the transmission spectrum (power vs. wavelength) for the first batch sample
    output_field = mzi(input_field)  # shape: (batch_size, N)
    # Compose the output complex field from the two channels:
    output_complex = torch.complex(output_field[0, 0], output_field[0, 1])
    # Compute power (magnitude squared) for each wavelength point
    power = torch.abs(output_complex)**2
    loss = linearity_loss(power)
    loss.backward()
    return loss

for epoch in range(num_epochs):
    loss = optimizer.step(closure)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6e}")
        # Assuming 'ring' is your instance of RingResonator
        for name, param in mzi.ring.named_parameters():
            print(f"{name}: {param.data.numpy()}")



# %% test loss function

import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_linearity_metrics(transmission):
    """
    Compute the derivative, the mask of positive slopes, and the average positive slope.
    
    transmission: tensor of shape (N,) representing the power vs. a normalized x-axis (0 to 1).
    
    Returns:
        dT: derivative (tensor of shape (N-1,))
        pos_mask: tensor (0 or 1) for points where dT is positive (shape (N-1,))
        avg_pos: scalar tensor representing the average positive slope
    """
    N = transmission.shape[0]
    # Assume the x-axis spans 0 to 1; dx = 1/N.
    dx = 1.0 / N
    dT = (transmission[1:] - transmission[:-1]) / dx  # approximate derivative, shape: (N-1,)
    
    # Create a mask for positive slopes
    pos_mask = (dT > 0).float()  # 1 if positive, 0 otherwise
    
    count = pos_mask.sum()
    if count.item() < 1:
        avg_pos = torch.tensor(0.0, dtype=torch.float32)
    else:
        avg_pos = (dT * pos_mask).sum() / count
        
    return dT, pos_mask, avg_pos

# For demonstration, let's create a dummy transmission curve.
# Replace this with your actual MZI transmission curve when available.
x_vals = torch.linspace(0, 1, power.shape[0])  # normalized x-axis from 0 to 1
# Create a dummy curve that has segments with positive slope.
# For example, a sine curve shifted upward.
# transmission = torch.sin(10 * torch.pi * x_vals) + 1.0
transmission = power

# Compute derivative, positive mask, and average positive slope.
dT, pos_mask, avg_pos = compute_linearity_metrics(transmission)

# Convert tensors to numpy for plotting.
x_plot = x_vals[:-1].numpy()       # x-values for derivative (N-1 points)
dT_np = dT.numpy()
mask_np = pos_mask.numpy()
avg_pos_val = avg_pos.item()

# Plot the derivative, marking positive-slope points in red, and draw a horizontal line at the average.
plt.figure(figsize=(10, 6))
plt.plot(x_plot, dT_np, label="Derivative dT/dx", color='blue')
# Mark only the points where the derivative is positive.
plt.scatter(x_plot[mask_np == 1], dT_np[mask_np == 1], color='red', label="Positive slope points")
plt.axhline(avg_pos_val, color='green', linestyle='--', label=f"Avg positive slope: {avg_pos_val:.3f}")
plt.xlabel("Normalized x")
plt.ylabel("dT/dx")
plt.title("Transmission Derivative with Positive Slope Mask and Average")
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
