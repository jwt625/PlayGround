"""
NMR Physics Module

Core physics functions for NMR simulations including:
- Rotation operations (Bloch sphere rotations)
- Phase evolution and integration
- Pulse sequence definitions
"""

import numpy as np


def rotate_x(x, y, z, angle):
    """
    Rotate vectors around x-axis by angle (in radians)
    
    Parameters:
    -----------
    x, y, z : array_like
        Input vector components
    angle : float
        Rotation angle in radians
        
    Returns:
    --------
    x_new, y_new, z_new : ndarray
        Rotated vector components
    """
    x_new = x
    y_new = y * np.cos(angle) - z * np.sin(angle)
    z_new = y * np.sin(angle) + z * np.cos(angle)
    return x_new, y_new, z_new


def rotate_y(x, y, z, angle):
    """
    Rotate vectors around y-axis by angle (in radians)
    
    Parameters:
    -----------
    x, y, z : array_like
        Input vector components
    angle : float
        Rotation angle in radians
        
    Returns:
    --------
    x_new, y_new, z_new : ndarray
        Rotated vector components
    """
    x_new = x * np.cos(angle) + z * np.sin(angle)
    y_new = y
    z_new = -x * np.sin(angle) + z * np.cos(angle)
    return x_new, y_new, z_new


def rotate_z(x, y, z, angle):
    """
    Rotate vectors around z-axis by angle (in radians)
    
    Parameters:
    -----------
    x, y, z : array_like
        Input vector components
    angle : float
        Rotation angle in radians
        
    Returns:
    --------
    x_new, y_new, z_new : ndarray
        Rotated vector components
    """
    x_new = x * np.cos(angle) - y * np.sin(angle)
    y_new = x * np.sin(angle) + y * np.cos(angle)
    z_new = z
    return x_new, y_new, z_new


def integrate_phase(frame_start, frame_end, frequencies_array, dt):
    """
    Integrate phase evolution over time with time-varying frequencies
    
    Parameters:
    -----------
    frame_start : int
        Starting frame index
    frame_end : int
        Ending frame index
    frequencies_array : ndarray
        Array of shape (n_frames, n_spins) containing frequencies at each time
    dt : float
        Time step between frames
        
    Returns:
    --------
    phases : ndarray
        Accumulated phase for each spin
    """
    n_spins = frequencies_array.shape[1]
    phases = np.zeros(n_spins)
    
    for i in range(frame_start, frame_end):
        if i >= len(frequencies_array):
            break
        phases += frequencies_array[i, :] * dt
    
    return phases


class PulseSequence:
    """
    Define a pulse sequence with timing and pulse parameters
    """
    
    def __init__(self, pulse_90_duration, pulse_180_duration, tau, extra_time=0):
        """
        Initialize Hahn Echo pulse sequence
        
        Parameters:
        -----------
        pulse_90_duration : float
            Duration of 90° pulse
        pulse_180_duration : float
            Duration of 180° pulse
        tau : float
            Time between pulses
        extra_time : float
            Extra time after echo for observation
        """
        self.pulse_90_duration = pulse_90_duration
        self.pulse_180_duration = pulse_180_duration
        self.tau = tau
        self.extra_time = extra_time
        
        # Calculate time segments
        self.t1_end = pulse_90_duration
        self.t2_end = self.t1_end + tau
        self.t3_end = self.t2_end + pulse_180_duration
        self.t4_end = self.t3_end + tau  # Echo time
        self.t_total = self.t4_end + extra_time
    
    def get_phase_label(self, time):
        """
        Get descriptive label for current phase of sequence
        
        Parameters:
        -----------
        time : float
            Current time in sequence
            
        Returns:
        --------
        label : str
            Description of current phase
        """
        if time <= self.t1_end:
            return '90° Pulse'
        elif time <= self.t2_end:
            return 'Free Precession'
        elif time <= self.t3_end:
            return '180° Pulse'
        elif time <= self.t4_end:
            return 'Echo'
        else:
            return 'Post-Echo Decay'

