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
        
        # Learnable parameters: scaling for kappa and delta
        self.kappa_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.delta_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.kappa_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.delta_shift = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

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
        kappa = (self.kappa_scale * self.kappa_normalized + self.kappa_shift) * 2 * torch.pi / self.wavelength_points
        delta_beta = (self.delta_scale * self.delta_beta_normalized + self.delta_shift) * 2 * torch.pi / self.wavelength_points
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


class SegmentedDirectionalCoupler(nn.Module):
    def __init__(self, wavelength_points, total_length, num_segments):
        """
        A segmented directional coupler composed of multiple layers of DirectionalCouplerCoupledMode.
        
        wavelength_points: array or list of wavelength points (e.g., in meters)
        total_length: total length of the coupler (meters)
        num_segments: number of segments (layers) in the coupler
        """
        super(SegmentedDirectionalCoupler, self).__init__()
        self.segments = nn.ModuleList([
            DirectionalCouplerCoupledMode(wavelength_points, L=total_length / num_segments)
            for _ in range(num_segments)
        ])
        
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
           where 4 channels represent [real1, imag1, real2, imag2]
           and N is the number of wavelength points.
        """
        for segment in self.segments:
            x = segment(x)  # Pass through each segment
        return x

