#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class YSplitter(nn.Module):
    def __init__(self):
        super(YSplitter, self).__init__()
        self.factor = 1 / np.sqrt(2)  # Factor for power conservation

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 2, N)
           where 2 channels represent [real, imag] of the input waveguide.
        Returns:
           output tensor of shape (batch_size, 4, N)
        """
        # Split the input into two waveguides and apply the factor
        real_part = x[:, 0, :] * self.factor  # Real part
        imag_part = x[:, 1, :] * self.factor  # Imaginary part
        
        # Create the output tensor
        output = torch.stack([
            real_part,  # Waveguide 1 real part
            imag_part,  # Waveguide 1 imag part
            real_part,  # Waveguide 2 real part (same as waveguide 1)
            imag_part   # Waveguide 2 imag part (same as waveguide 1)
        ], dim=1)  # shape: (batch_size, 4, N)

        return output


class YCombiner(nn.Module):
    def __init__(self):
        super(YCombiner, self).__init__()
        self.factor = 1 / np.sqrt(2)  # Factor for power conservation

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 4, N)
           where 4 channels represent [real1, imag1, real2, imag2].
        Returns:
           output tensor of shape (batch_size, 2, N)
        """
        # Average the two waveguides and apply the factor
        real_part = (x[:, 0, :] + x[:, 2, :]) * self.factor  # Average real parts
        imag_part = (x[:, 1, :] + x[:, 3, :]) * self.factor  # Average imag parts
        
        # Create the output tensor
        output = torch.stack([
            real_part,  # Combined real part
            imag_part   # Combined imag part
        ], dim=1)  # shape: (batch_size, 2, N)

        return output

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

