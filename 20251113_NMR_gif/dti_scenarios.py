"""
DTI Scenarios Module

Defines diffusion tensor scenarios and gradient encoding schemes for DTI simulations.
"""

import numpy as np


def rotation_matrix_x(theta):
    """Rotation matrix around x-axis by angle theta (radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(theta):
    """Rotation matrix around y-axis by angle theta (radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(theta):
    """Rotation matrix around z-axis by angle theta (radians)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotate_tensor(D, R):
    """
    Rotate diffusion tensor D by rotation matrix R
    D' = R * D * R^T
    """
    return R @ D @ R.T


class DTIScenario:
    """
    Define a DTI scenario with diffusion tensor and metadata
    """
    
    def __init__(self, name, diffusion_tensor, description=""):
        """
        Parameters:
        -----------
        name : str
            Scenario name
        diffusion_tensor : ndarray
            3x3 diffusion tensor
        description : str
            Description of the scenario
        """
        self.name = name
        self.D = np.array(diffusion_tensor)
        self.description = description
        
        # Compute eigendecomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.D)
        # Sort by eigenvalue (descending)
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        # DTI metrics
        self.lambda1, self.lambda2, self.lambda3 = self.eigenvalues
        self.MD = np.mean(self.eigenvalues)
        self.FA = self._compute_FA()
        self.principal_direction = self.eigenvectors[:, 0]
    
    def _compute_FA(self):
        """Compute Fractional Anisotropy"""
        MD = self.MD
        numerator = np.sqrt(((self.lambda1 - MD)**2 + 
                            (self.lambda2 - MD)**2 + 
                            (self.lambda3 - MD)**2))
        denominator = np.sqrt(self.lambda1**2 + self.lambda2**2 + self.lambda3**2)
        if denominator == 0:
            return 0.0
        return np.sqrt(3/2) * numerator / denominator
    
    def get_ADC(self, gradient_direction):
        """
        Compute apparent diffusion coefficient for a given gradient direction
        ADC = g^T * D * g
        
        Parameters:
        -----------
        gradient_direction : array_like
            Gradient direction (will be normalized)
        
        Returns:
        --------
        ADC : float
            Apparent diffusion coefficient
        """
        g = np.array(gradient_direction)
        g = g / np.linalg.norm(g)
        return g @ self.D @ g
    
    def __str__(self):
        s = f"DTI Scenario: {self.name}\n"
        s += f"Description: {self.description}\n"
        s += f"Diffusion Tensor:\n{self.D}\n"
        s += f"Eigenvalues: λ1={self.lambda1:.4f}, λ2={self.lambda2:.4f}, λ3={self.lambda3:.4f}\n"
        s += f"MD = {self.MD:.4f}, FA = {self.FA:.4f}\n"
        s += f"Principal direction: {self.principal_direction}\n"
        return s


def get_standard_scenarios():
    """
    Get the 4 standard DTI scenarios
    
    Returns:
    --------
    scenarios : dict
        Dictionary of DTIScenario objects
    """
    scenarios = {}
    
    # Case 1: Isotropic
    D_iso = np.diag([0.05, 0.05, 0.05])
    scenarios['isotropic'] = DTIScenario(
        name="Isotropic",
        diffusion_tensor=D_iso,
        description="Free water, CSF - equal diffusion in all directions"
    )
    
    # Case 2: Anisotropic - Z fiber
    D_z = np.diag([0.01, 0.01, 0.10])
    scenarios['z_fiber'] = DTIScenario(
        name="Z-Fiber",
        diffusion_tensor=D_z,
        description="White matter tract aligned with z-axis"
    )
    
    # Case 3: Anisotropic - X fiber
    D_x = np.diag([0.10, 0.01, 0.01])
    scenarios['x_fiber'] = DTIScenario(
        name="X-Fiber",
        diffusion_tensor=D_x,
        description="White matter tract aligned with x-axis"
    )
    
    # Case 4: Anisotropic - Tilted fiber (45° in XZ plane)
    theta = np.pi / 4
    R_y = rotation_matrix_y(theta)
    D_tilted = rotate_tensor(D_z, R_y)
    scenarios['tilted_fiber'] = DTIScenario(
        name="Tilted-Fiber",
        diffusion_tensor=D_tilted,
        description="White matter tract at 45° in XZ plane"
    )
    
    return scenarios


def get_gradient_scheme(scheme='6dir'):
    """
    Get gradient encoding scheme
    
    Parameters:
    -----------
    scheme : str
        '6dir' - 6 orthogonal directions (minimum for tensor fitting)
        '12dir' - 12 directions
        '30dir' - 30 directions (not implemented yet)
    
    Returns:
    --------
    gradients : ndarray
        Array of shape (N, 3) with gradient directions
    labels : list
        List of labels for each gradient
    """
    if scheme == '6dir':
        gradients = np.array([
            [1, 0, 0],      # Gx
            [0, 1, 0],      # Gy
            [0, 0, 1],      # Gz
            [1, 1, 0],      # G_xy
            [1, 0, 1],      # G_xz
            [0, 1, 1],      # G_yz
        ], dtype=float)
        labels = ['Gx', 'Gy', 'Gz', 'Gxy', 'Gxz', 'Gyz']
        
    elif scheme == '12dir':
        # 12 directions: 6 orthogonal + 6 more diagonal
        gradients = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            [1, -1, 0], [0, 1, -1]
        ], dtype=float)
        labels = [f'G{i+1}' for i in range(12)]
    else:
        raise ValueError(f"Unknown gradient scheme: {scheme}")
    
    # Normalize all gradients
    gradients = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)
    
    return gradients, labels

