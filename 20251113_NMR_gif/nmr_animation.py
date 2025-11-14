"""
NMR Animation Module

Animation creation and export functionality for NMR simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import nmr_visualization as nmr_viz


class HahnEchoAnimator:
    """
    Create animations for Hahn Echo simulations
    """
    
    def __init__(self, simulator, figsize=(12, 8), dpi=80):
        """
        Initialize animator
        
        Parameters:
        -----------
        simulator : HahnEchoSimulator
            Simulator object
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure DPI
        """
        self.simulator = simulator
        self.figsize = figsize
        self.dpi = dpi
        
        # Storage for magnetization history
        self.mag_history_x = []
        self.mag_history_y = []
        self.mag_history_magnitude = []
        self.time_history = []
        
        # Figure and axes
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        
    def create_three_panel_figure(self):
        """
        Create figure with 3 panels: two 3D on top, one 2D below
        
        Returns:
        --------
        fig : matplotlib figure
        ax1, ax2, ax3 : matplotlib axes
        """
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax1 = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax2 = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.ax3 = self.fig.add_subplot(2, 1, 2)
        
        return self.fig, self.ax1, self.ax2, self.ax3
    
    def init_animation(self):
        """Initialize animation"""
        return []
    
    def animate_frame(self, frame):
        """
        Animate a single frame with diffusion visualization
        
        Parameters:
        -----------
        frame : int
            Frame number
            
        Returns:
        --------
        artists : list
            List of artists (for blitting)
        """
        self.ax1.clear()
        self.ax2.clear()
        
        # Left plot: With diffusion (real space positions)
        nmr_viz.draw_bloch_sphere(self.ax1)
        x1, y1, z1, label = self.simulator.evolve_spins(frame, with_diffusion=True)
        
        # Calculate net magnetization
        net_mag_x, net_mag_y, net_mag_magnitude = self.simulator.compute_net_magnetization(x1, y1, z1)
        
        # Store history
        self.mag_history_x.append(net_mag_x)
        self.mag_history_y.append(net_mag_y)
        self.mag_history_magnitude.append(net_mag_magnitude)
        self.time_history.append(self.simulator.t[frame])
        
        # Get real space positions for this frame
        if self.simulator.diffusion_sim is not None:
            positions_real = self.simulator.diffusion_sim.positions[frame, :, :]  # Shape: (n_spins, 3)

            # Draw spins with their real space origins
            # Scale positions for visualization (0.3 scale factor)
            origins = positions_real * 0.3
            nmr_viz.plot_spins(self.ax1, x1, y1, z1, color='red', alpha=0.6, origins=origins)
        else:
            nmr_viz.plot_spins(self.ax1, x1, y1, z1, color='red', alpha=0.6)
        
        nmr_viz.draw_axes(self.ax1)
        nmr_viz.setup_bloch_axis(self.ax1, title='With Diffusion\n(Real Space Origins)', 
                                 elev=20, azim=45)
        
        # Right plot: Same diffusion but overlapping origins
        nmr_viz.draw_bloch_sphere(self.ax2)
        x2, y2, z2, _ = self.simulator.evolve_spins(frame, with_diffusion=True)
        nmr_viz.plot_spins(self.ax2, x2, y2, z2, color='blue', alpha=0.6)
        nmr_viz.draw_axes(self.ax2)
        nmr_viz.setup_bloch_axis(self.ax2, title='With Diffusion\n(Overlapping Origins)', 
                                 elev=20, azim=45)
        
        # Add overall title with parameters
        if self.simulator.diffusion_sim is not None:
            D = self.simulator.diffusion_sim.diffusion_coefficient
            G = self.simulator.diffusion_sim.gradient_strength
            title_text = f'Hahn Echo with Diffusion | D={D:.3f}, G={G:.3f} | {label}'
        else:
            title_text = f'Hahn Echo | {label}'
        self.fig.suptitle(title_text, fontsize=12, fontweight='bold', y=0.98)
        
        # Bottom plot: Net magnetization over time
        nmr_viz.plot_magnetization_history(
            self.ax3, 
            self.time_history, 
            self.mag_history_x, 
            self.mag_history_y,
            self.mag_history_magnitude,
            self.simulator.pulse_sequence,
            self.simulator.pulse_sequence.t_total
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        return []
    
    def create_animation(self, fps=15, interval=50):
        """
        Create animation
        
        Parameters:
        -----------
        fps : int
            Frames per second for output
        interval : int
            Delay between frames in milliseconds
            
        Returns:
        --------
        anim : FuncAnimation
            Animation object
        """
        if self.fig is None:
            self.create_three_panel_figure()
        
        # Reset history
        self.mag_history_x = []
        self.mag_history_y = []
        self.mag_history_magnitude = []
        self.time_history = []
        
        anim = FuncAnimation(self.fig, self.animate_frame, 
                           init_func=self.init_animation,
                           frames=self.simulator.n_frames,
                           interval=interval, blit=False)
        
        return anim
    
    def save_animation(self, filename, fps=15):
        """
        Save animation as GIF
        
        Parameters:
        -----------
        filename : str
            Output filename
        fps : int
            Frames per second
        """
        anim = self.create_animation(fps=fps)
        
        print(f"Saving animation to '{filename}'...")
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"Animation saved successfully!")
        
        plt.close()

