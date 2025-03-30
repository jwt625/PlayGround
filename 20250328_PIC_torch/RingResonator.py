import torch
import torch.nn as nn
import numpy as np


class RingResonatorFP(nn.Module):
    def __init__(self, wavelength_points, L=10, r=0.9, a=0.99, phase_shift=0.0):
        """
        A ring resonator modeled using the FP cavity analytical solution.
        
        In this mapping, the FP cavity reflection (H) is taken as the ring resonator 
        through-port, and the FP transmission is treated as the internal loss.
        
        Parameters:
          wavelength_points : array-like, wavelengths in meters.
          n_eff             : effective index of the ring.
          L                 : round-trip length of the ring (meters).
          r                 : self-coupling coefficient (amplitude reflection at the coupler).
          a                 : round-trip amplitude transmission (internal loss factor).
          phase_shift       : additional phase shift to apply to the resonances.
        """
        super(RingResonatorFP, self).__init__()
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float32))  # in um
        # Optionally make r and a learnable:
        self.r = nn.Parameter(torch.tensor(r, dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.phase_shift = nn.Parameter(torch.tensor(phase_shift, dtype=torch.float32))  # New phase parameter
        
        # Register wavelength points as a buffer (non-trainable)
        self.register_buffer('wavelength_points', 
                             torch.tensor(wavelength_points, dtype=torch.float32))
    
    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 2, N)
           where the 2 channels represent the real and imaginary parts of the field,
           and N is the number of wavelength points.
        """
        # Compute the phase shift for each wavelength:
        # θ(λ) = 2π n_eff L / λ
        L_meter = self.L * 1e-6  # convert to meters
        theta = 2 * torch.pi * L_meter / self.wavelength_points  # shape: (N,)
        
        # Compute the transfer function H(λ) using the FP cavity reflection formula:
        # Incorporate the additional phase shift
        H = (self.r - self.a * torch.exp(-1j * (theta + self.phase_shift))) / (1 - self.r * self.a * torch.exp(-1j * (theta + self.phase_shift)))
        # H has shape: (N,)
        
        # Convert input field into a complex tensor.
        # x is assumed to have 2 channels: channel 0 = real, channel 1 = imag.
        input_complex = torch.complex(x[:, 0], x[:, 1])  # shape: (batch_size, N)
        
        # Apply the transfer function (broadcast H over the batch dimension)
        output_complex = input_complex * H.unsqueeze(0)  # shape: (batch_size, N)
        
        # Return the output as two channels: real and imaginary parts.
        output_real = output_complex.real
        output_imag = output_complex.imag
        return torch.stack([output_real, output_imag], dim=1)  # shape: (batch_size, 2, N)
