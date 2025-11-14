"""
NMR Simulator Module

Main simulation engine for NMR experiments including:
- Hahn Echo sequence evolution
- Spin state calculation
- Integration with diffusion effects
"""

import numpy as np
from nmr_physics import rotate_x, rotate_y, integrate_phase, PulseSequence


class HahnEchoSimulator:
    """
    Simulate Hahn Echo pulse sequence with optional diffusion
    """
    
    def __init__(self, n_spins, pulse_sequence, diffusion_sim=None):
        """
        Initialize Hahn Echo simulator
        
        Parameters:
        -----------
        n_spins : int
            Number of spins
        pulse_sequence : PulseSequence
            Pulse sequence object with timing
        diffusion_sim : DiffusionSimulator, optional
            Diffusion simulator object. If None, no diffusion effects
        """
        self.n_spins = n_spins
        self.pulse_sequence = pulse_sequence
        self.diffusion_sim = diffusion_sim
        
        # Time array
        self.n_frames = None
        self.t = None
        self.dt = None
        
    def setup_time_array(self, n_frames):
        """
        Setup time array for simulation
        
        Parameters:
        -----------
        n_frames : int
            Number of frames in simulation
        """
        self.n_frames = n_frames
        self.t = np.linspace(0, self.pulse_sequence.t_total, n_frames)
        self.dt = self.t[1] - self.t[0] if len(self.t) > 1 else 1
        
    def evolve_spins(self, time_idx, with_diffusion=True):
        """
        Calculate spin positions during Hahn echo sequence
        
        Parameters:
        -----------
        time_idx : int
            Current time frame index
        with_diffusion : bool
            Whether to include diffusion effects
            
        Returns:
        --------
        x, y, z : ndarray
            Spin vector components
        phase_label : str
            Description of current sequence phase
        """
        if self.t is None:
            raise ValueError("Must call setup_time_array first")
            
        time = self.t[time_idx]
        ps = self.pulse_sequence  # Shorthand
        
        # Initialize spins along +z (equilibrium)
        x = np.zeros(self.n_spins)
        y = np.zeros(self.n_spins)
        z = np.ones(self.n_spins)
        
        # Get phase label
        phase_label = ps.get_phase_label(time)
        
        # Segment 1: 90° pulse around x-axis
        if time <= ps.t1_end:
            angle = (time / ps.pulse_90_duration) * (np.pi / 2)
            x, y, z = rotate_x(x, y, z, angle)
        
        # Segment 2: Free precession
        elif time <= ps.t2_end:
            phases = self._compute_phases(int(ps.t1_end/self.dt), time_idx, 
                                         with_diffusion)
            x = -np.sin(phases)
            y = -np.cos(phases)
            z = np.zeros(self.n_spins)
        
        # Segment 3: 180° pulse around y-axis
        elif time <= ps.t3_end:
            # Get positions just before 180° pulse
            phases = self._compute_phases(int(ps.t1_end/self.dt), 
                                         int(ps.t2_end/self.dt), 
                                         with_diffusion)
            x_before = -np.sin(phases)
            y_before = -np.cos(phases)
            z_before = np.zeros(self.n_spins)
            
            # Apply 180° rotation around y-axis
            pulse_progress = (time - ps.t2_end) / ps.pulse_180_duration
            angle = pulse_progress * np.pi
            x, y, z = rotate_y(x_before, y_before, z_before, angle)
        
        # Segment 4: Refocusing and beyond
        else:
            # Phase evolution: first half + second half (with sign flip from 180° pulse)
            phases_first = self._compute_phases(int(ps.t1_end/self.dt), 
                                               int(ps.t2_end/self.dt), 
                                               with_diffusion)
            phases_second = -self._compute_phases(int(ps.t3_end/self.dt), 
                                                  time_idx, 
                                                  with_diffusion)
            phases = phases_first + phases_second
            
            x = -np.sin(phases)
            y = -np.cos(phases)
            z = np.zeros(self.n_spins)
        
        return x, y, z, phase_label
    
    def _compute_phases(self, frame_start, frame_end, with_diffusion):
        """
        Compute phase accumulation between two time points
        
        Parameters:
        -----------
        frame_start : int
            Starting frame
        frame_end : int
            Ending frame
        with_diffusion : bool
            Whether to include diffusion effects
            
        Returns:
        --------
        phases : ndarray
            Phase for each spin
        """
        if with_diffusion and self.diffusion_sim is not None:
            # Use time-varying frequencies from diffusion
            if self.diffusion_sim.frequencies_array is None:
                raise ValueError("Must compute frequencies in diffusion simulator first")
            phases = integrate_phase(frame_start, frame_end, 
                                    self.diffusion_sim.frequencies_array, 
                                    self.dt)
        else:
            # No diffusion: constant frequencies
            if self.diffusion_sim is not None and self.diffusion_sim.base_frequencies is not None:
                base_freq = self.diffusion_sim.base_frequencies
            else:
                # Default: random frequencies
                base_freq = np.random.randn(self.n_spins) * 0.25
            
            time_elapsed = (frame_end - frame_start) * self.dt
            phases = base_freq * time_elapsed
        
        return phases
    
    def compute_net_magnetization(self, x, y, z):
        """
        Compute net magnetization from individual spins
        
        Parameters:
        -----------
        x, y, z : ndarray
            Spin vector components
            
        Returns:
        --------
        net_mag_x, net_mag_y, net_mag_magnitude : float
            Net magnetization components and magnitude
        """
        net_mag_x = np.sum(x) / self.n_spins
        net_mag_y = np.sum(y) / self.n_spins
        net_mag_magnitude = np.sqrt(net_mag_x**2 + net_mag_y**2)
        
        return net_mag_x, net_mag_y, net_mag_magnitude

