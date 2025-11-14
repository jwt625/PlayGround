"""
NMR Visualization Module

Visualization functions for NMR simulations including:
- Bloch sphere rendering
- Spin vector plotting
- Magnetization time curves
- Multi-panel layouts
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_bloch_sphere(ax, resolution='medium'):
    """
    Draw Bloch sphere wireframe
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to draw on
    resolution : str
        'low', 'medium', or 'high' - controls wireframe density
    """
    res_map = {'low': (20, 15), 'medium': (30, 20), 'high': (50, 50)}
    u_res, v_res = res_map.get(resolution, (30, 20))
    
    u = np.linspace(0, 2 * np.pi, u_res)
    v = np.linspace(0, np.pi, v_res)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    ax.plot_wireframe(x, y, z, alpha=0.15, color='gray', linewidth=0.3)


def draw_axes(ax, length=1.3, label_offset=1.4):
    """
    Draw coordinate axes on Bloch sphere
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to draw on
    length : float
        Length of axis lines
    label_offset : float
        Position of axis labels
    """
    ax.plot([0, length], [0, 0], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax.plot([0, 0], [0, length], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax.plot([0, 0], [0, 0], [0, length], 'k-', linewidth=1.5, alpha=0.5)
    ax.text(label_offset, 0, 0, 'X', fontsize=10, fontweight='bold')
    ax.text(0, label_offset, 0, 'Y', fontsize=10, fontweight='bold')
    ax.text(0, 0, label_offset, 'Z', fontsize=10, fontweight='bold')


def plot_spins(ax, x, y, z, color='red', alpha=0.6, show_arrows=True, 
               show_points=True, origins=None):
    """
    Plot spin vectors on Bloch sphere
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to draw on
    x, y, z : array_like
        Spin vector components
    color : str
        Color of spins
    alpha : float
        Transparency
    show_arrows : bool
        Whether to show arrows
    show_points : bool
        Whether to show endpoint dots
    origins : array_like, optional
        Custom origins for spins (n_spins, 3). If None, all from origin
    """
    n_spins = len(x)
    
    if origins is None:
        origins = np.zeros((n_spins, 3))
    
    if show_arrows:
        for i in range(n_spins):
            ax.quiver(origins[i, 0], origins[i, 1], origins[i, 2],
                     x[i], y[i], z[i],
                     color=color, alpha=alpha, arrow_length_ratio=0.15, linewidth=1.5)
    
    if show_points:
        ax.scatter(x + origins[:, 0], y + origins[:, 1], z + origins[:, 2], 
                  c=color, s=30, alpha=alpha+0.1)


def setup_bloch_axis(ax, title='', elev=20, azim=45):
    """
    Setup 3D axis for Bloch sphere visualization
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to setup
    title : str
        Title for the plot
    elev : float
        Elevation viewing angle
    azim : float
        Azimuthal viewing angle
    """
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)


def plot_magnetization_history(ax, time_history, mag_history_x, mag_history_y, 
                               mag_history_magnitude, pulse_sequence, t_total):
    """
    Plot net magnetization over time
    
    Parameters:
    -----------
    ax : matplotlib 2D axis
        Axis to plot on
    time_history : list
        Time points
    mag_history_x, mag_history_y : list
        Magnetization components
    mag_history_magnitude : list
        Magnetization magnitude
    pulse_sequence : PulseSequence
        Pulse sequence object with timing information
    t_total : float
        Total simulation time
    """
    ax.clear()
    ax.plot(time_history, mag_history_x, 'r-', label='Mx (transverse)', linewidth=2)
    ax.plot(time_history, mag_history_y, 'b-', label='My (transverse)', linewidth=2)
    ax.plot(time_history, mag_history_magnitude, 'k-', label='|M_xy| (magnitude)', linewidth=2.5)
    
    # Mark pulse times
    ax.axvline(pulse_sequence.t1_end, color='green', linestyle='--', 
              alpha=0.5, linewidth=1.5, label='90° pulse end')
    ax.axvline(pulse_sequence.t2_end, color='orange', linestyle='--', 
              alpha=0.5, linewidth=1.5, label='180° pulse start')
    ax.axvline(pulse_sequence.t3_end, color='purple', linestyle='--', 
              alpha=0.5, linewidth=1.5, label='180° pulse end')
    ax.axvline(pulse_sequence.t4_end, color='red', linestyle='--', 
              alpha=0.5, linewidth=1.5, label='Echo time')
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Net Magnetization', fontsize=11)
    ax.set_title('Sum of All Spins (Net Magnetization)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t_total])
    ax.set_ylim([-1.1, 1.1])

