"""
Diffusion Module

Handles diffusion simulation including:
- Random walk generation (Brownian motion)
- Isotropic and anisotropic diffusion
- Gradient effects on spin frequencies
- Diffusion tensor operations
"""

import numpy as np


class DiffusionSimulator:
    """
    Simulate molecular diffusion and its effects on NMR signals
    """
    
    def __init__(self, n_spins, n_frames, dt, diffusion_coefficient=0.05, 
                 gradient_strength=0.01, seed=42):
        """
        Initialize diffusion simulator
        
        Parameters:
        -----------
        n_spins : int
            Number of spins to simulate
        n_frames : int
            Number of time frames
        dt : float
            Time step between frames
        diffusion_coefficient : float
            Diffusion coefficient (isotropic case)
        gradient_strength : float
            Magnetic field gradient strength (frequency change per unit position)
        seed : int
            Random seed for reproducibility
        """
        self.n_spins = n_spins
        self.n_frames = n_frames
        self.dt = dt
        self.diffusion_coefficient = diffusion_coefficient
        self.gradient_strength = gradient_strength
        
        np.random.seed(seed)
        
        # Initialize positions
        self.positions = None
        self.base_frequencies = None
        self.frequencies_array = None
    
    def generate_random_walk_1d(self, initial_spread=0.1):
        """
        Generate 1D random walk trajectories (diffusion along z-axis)
        
        Parameters:
        -----------
        initial_spread : float
            Standard deviation of initial position distribution
            
        Returns:
        --------
        positions : ndarray
            Array of shape (n_frames, n_spins) with z-positions
        """
        positions = np.zeros((self.n_frames, self.n_spins))
        positions[0, :] = np.random.randn(self.n_spins) * initial_spread
        
        # Random walk
        for i in range(1, self.n_frames):
            step = np.random.randn(self.n_spins) * np.sqrt(2 * self.diffusion_coefficient * self.dt)
            positions[i, :] = positions[i-1, :] + step
        
        self.positions = positions
        return positions
    
    def generate_random_walk_3d(self, diffusion_tensor=None, initial_spread=0.1):
        """
        Generate 3D random walk trajectories with optional anisotropic diffusion
        
        Parameters:
        -----------
        diffusion_tensor : ndarray, optional
            3x3 diffusion tensor matrix. If None, uses isotropic diffusion
        initial_spread : float
            Standard deviation of initial position distribution
            
        Returns:
        --------
        positions : ndarray
            Array of shape (n_frames, n_spins, 3) with xyz-positions
        """
        positions = np.zeros((self.n_frames, self.n_spins, 3))
        positions[0, :, :] = np.random.randn(self.n_spins, 3) * initial_spread
        
        if diffusion_tensor is None:
            # Isotropic diffusion
            diffusion_tensor = np.eye(3) * self.diffusion_coefficient
        
        # Compute Cholesky decomposition for correlated random walk
        # D = L * L^T, so step = L * randn gives correct covariance
        L = np.linalg.cholesky(2 * diffusion_tensor * self.dt)
        
        # Random walk
        for i in range(1, self.n_frames):
            random_steps = np.random.randn(self.n_spins, 3)
            # Apply diffusion tensor via matrix multiplication
            correlated_steps = random_steps @ L.T
            positions[i, :, :] = positions[i-1, :, :] + correlated_steps
        
        self.positions = positions
        return positions
    
    def set_base_frequencies(self, frequency_spread=0.25):
        """
        Set base frequencies for spins (chemical shift, B0 inhomogeneity, etc.)
        
        Parameters:
        -----------
        frequency_spread : float
            Standard deviation of frequency distribution
            
        Returns:
        --------
        base_frequencies : ndarray
            Array of base frequencies for each spin
        """
        self.base_frequencies = np.random.randn(self.n_spins) * frequency_spread
        return self.base_frequencies
    
    def compute_frequencies_with_gradient(self, gradient_direction='z'):
        """
        Compute time-varying frequencies including gradient effects
        
        Parameters:
        -----------
        gradient_direction : str or ndarray
            Direction of gradient ('x', 'y', 'z') or 3D vector
            
        Returns:
        --------
        frequencies_array : ndarray
            Array of shape (n_frames, n_spins) with frequencies at each time
        """
        if self.positions is None:
            raise ValueError("Must generate positions first")
        if self.base_frequencies is None:
            self.set_base_frequencies()
        
        frequencies_array = np.zeros((self.n_frames, self.n_spins))
        
        # Handle 1D case (positions is 2D)
        if len(self.positions.shape) == 2:
            # Assume z-direction
            for i in range(self.n_frames):
                frequencies_array[i, :] = (self.base_frequencies + 
                                          self.gradient_strength * self.positions[i, :])
        else:
            # 3D case
            if isinstance(gradient_direction, str):
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                axis = axis_map[gradient_direction.lower()]
                for i in range(self.n_frames):
                    frequencies_array[i, :] = (self.base_frequencies + 
                                              self.gradient_strength * self.positions[i, :, axis])
            else:
                # Custom gradient direction (3D vector)
                gradient_direction = np.array(gradient_direction)
                gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)
                for i in range(self.n_frames):
                    # Project position onto gradient direction
                    projection = np.sum(self.positions[i, :, :] * gradient_direction, axis=1)
                    frequencies_array[i, :] = (self.base_frequencies + 
                                              self.gradient_strength * projection)
        
        self.frequencies_array = frequencies_array
        return frequencies_array

