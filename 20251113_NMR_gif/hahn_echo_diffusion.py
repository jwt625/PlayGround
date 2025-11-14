import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_spins = 60  # Number of spins
n_frames = 150  # Total frames for animation
pulse_90_duration = 5  # Duration of 90° pulse in frames
pulse_180_duration = 6  # Duration of 180° pulse in frames
tau = 50  # Time between pulses (in frame units)

# Diffusion parameters
diffusion_coefficient = 0.005  # Diffusion rate
gradient_strength = 0.01  # Gradient strength (frequency change per unit z)

# Define time segments
t1_end = pulse_90_duration
t2_end = t1_end + tau
t3_end = t2_end + pulse_180_duration
t4_end = t3_end + tau

# Time points
t = np.linspace(0, t4_end, n_frames)
dt = t[1] - t[0] if len(t) > 1 else 1

# Initialize spin positions in real space (z-coordinates)
np.random.seed(42)
z_positions = np.zeros((n_frames, n_spins))  # z-position for each spin at each time
z_positions[0, :] = np.random.randn(n_spins) * 0.1  # Small initial spread

# Generate diffusion trajectories (random walk)
for i in range(1, n_frames):
    z_positions[i, :] = z_positions[i-1, :] + np.random.randn(n_spins) * np.sqrt(2 * diffusion_coefficient * dt)

# Base frequencies (Gaussian distribution, 5x wider spread)
base_frequencies = np.random.randn(n_spins) * 0.25  # 5x wider: 0.05 * 5 = 0.25 std dev

def rotate_x(x, y, z, angle):
    """Rotate vectors around x-axis by angle (in radians)"""
    x_new = x
    y_new = y * np.cos(angle) - z * np.sin(angle)
    z_new = y * np.sin(angle) + z * np.cos(angle)
    return x_new, y_new, z_new

def rotate_y(x, y, z, angle):
    """Rotate vectors around y-axis by angle (in radians)"""
    x_new = x * np.cos(angle) + z * np.sin(angle)
    y_new = y
    z_new = -x * np.sin(angle) + z * np.cos(angle)
    return x_new, y_new, z_new

def integrate_phase(frame_start, frame_end, z_pos_array):
    """Integrate phase evolution considering time-varying z-positions"""
    phases = np.zeros(n_spins)
    for i in range(frame_start, frame_end):
        if i >= n_frames:
            break
        # Frequency at each time depends on z-position
        freq = base_frequencies + gradient_strength * z_pos_array[i, :]
        phases += freq * dt
    return phases

def hahn_echo_evolution(time_idx, with_diffusion=True):
    """
    Calculate spin positions during Hahn echo sequence
    with_diffusion: if True, use actual z-positions; if False, use z=0 for all
    """
    time = t[time_idx]
    
    # Initialize spins along +z (equilibrium)
    x = np.zeros(n_spins)
    y = np.zeros(n_spins)
    z = np.ones(n_spins)
    
    # Segment 1: 90° pulse around x-axis
    if time <= t1_end:
        angle = (time / pulse_90_duration) * (np.pi / 2)
        x, y, z = rotate_x(x, y, z, angle)
        phase_label = f'90° Pulse\n{np.degrees(angle):.0f}°'
    
    # Segment 2: Free precession
    elif time <= t2_end:
        # Integrate phase from end of pulse to current time
        if with_diffusion:
            phases = integrate_phase(int(t1_end/dt), time_idx, z_positions)
        else:
            # No diffusion: constant frequencies
            time_since_pulse = time - t1_end
            phases = base_frequencies * time_since_pulse
        
        x = -np.sin(phases)
        y = -np.cos(phases)
        z = np.zeros(n_spins)
        phase_label = f'Free Precession\n{time-t1_end:.0f}/{tau:.0f}'
    
    # Segment 3: 180° pulse around y-axis
    elif time <= t3_end:
        # Get positions just before 180° pulse
        if with_diffusion:
            phases = integrate_phase(int(t1_end/dt), int(t2_end/dt), z_positions)
        else:
            phases = base_frequencies * tau
        
        x_before = -np.sin(phases)
        y_before = -np.cos(phases)
        z_before = np.zeros(n_spins)
        
        pulse_progress = (time - t2_end) / pulse_180_duration
        angle = pulse_progress * np.pi
        x, y, z = rotate_y(x_before, y_before, z_before, angle)
        phase_label = f'180° Pulse\n{np.degrees(angle):.0f}°'
    
    # Segment 4: Refocusing
    else:
        # Phase evolution: first half + second half (with sign flip from 180° pulse)
        if with_diffusion:
            # First half: 0 to tau
            phases_first = integrate_phase(int(t1_end/dt), int(t2_end/dt), z_positions)
            # Second half: tau to current (with opposite sign due to 180° pulse)
            phases_second = -integrate_phase(int(t3_end/dt), time_idx, z_positions)
            phases = phases_first + phases_second
        else:
            time_after_180 = time - t3_end
            phases = base_frequencies * (tau - time_after_180)
        
        x = -np.sin(phases)
        y = -np.cos(phases)
        z = np.zeros(n_spins)
        phase_label = f'Echo\n{time-t3_end:.0f}/{tau:.0f}'
    
    return x, y, z, phase_label

def draw_bloch_sphere(ax):
    """Draw simplified Bloch sphere"""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    ax.plot_wireframe(x, y, z, alpha=0.15, color='gray', linewidth=0.3)

# Storage for magnetization history
mag_history_x = []
mag_history_y = []
mag_history_magnitude = []
time_history = []

# Create figure with 3 subplots: two 3D on top, one 2D below
fig = plt.figure(figsize=(12, 8), dpi=80)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 1, 2)  # 2D plot below spanning both columns

def init():
    """Initialize animation"""
    return []

def animate(frame):
    """Animation function"""
    ax1.clear()
    ax2.clear()

    # Left plot: With diffusion (real space positions)
    draw_bloch_sphere(ax1)
    x1, y1, z1, label = hahn_echo_evolution(frame, with_diffusion=True)

    # Calculate net magnetization (sum of all spins)
    net_mag_x = np.sum(x1) / n_spins
    net_mag_y = np.sum(y1) / n_spins
    net_mag_magnitude = np.sqrt(net_mag_x**2 + net_mag_y**2)

    # Store history
    mag_history_x.append(net_mag_x)
    mag_history_y.append(net_mag_y)
    mag_history_magnitude.append(net_mag_magnitude)
    time_history.append(t[frame])

    # Get real space z-positions for this frame
    z_real = z_positions[frame, :]

    # Draw spins with their real space origins
    for i in range(n_spins):
        # Origin is offset by real space z-position (scaled for visibility)
        origin_offset = z_real[i] * 0.3  # Scale factor for visualization
        ax1.quiver(0, 0, origin_offset, x1[i], y1[i], z1[i],
                  color='red', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5)

    ax1.scatter(x1, y1, z1 + z_real * 0.3, c='red', s=30, alpha=0.7)

    # Draw axes
    ax1.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax1.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax1.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1.5, alpha=0.5)

    ax1.set_xlim([-1.2, 1.2])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_zlim([-1.2, 1.2])
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title('With Diffusion\n(Real Space Origins)', fontsize=11, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Right plot: Same diffusion but overlapping origins (just visualization difference)
    draw_bloch_sphere(ax2)
    x2, y2, z2, _ = hahn_echo_evolution(frame, with_diffusion=True)

    # Draw spins from common origin
    for i in range(n_spins):
        ax2.quiver(0, 0, 0, x2[i], y2[i], z2[i],
                  color='blue', alpha=0.6, arrow_length_ratio=0.15, linewidth=1.5)

    ax2.scatter(x2, y2, z2, c='blue', s=30, alpha=0.7)

    # Draw axes
    ax2.plot([0, 1.2], [0, 0], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax2.plot([0, 0], [0, 1.2], [0, 0], 'k-', linewidth=1.5, alpha=0.5)
    ax2.plot([0, 0], [0, 0], [0, 1.2], 'k-', linewidth=1.5, alpha=0.5)

    ax2.set_xlim([-1.2, 1.2])
    ax2.set_ylim([-1.2, 1.2])
    ax2.set_zlim([-1.2, 1.2])
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    ax2.set_title('With Diffusion\n(Overlapping Origins)', fontsize=11, fontweight='bold')
    ax2.view_init(elev=20, azim=45)

    # Add phase label
    fig.text(0.5, 0.95, label, ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Bottom plot: Net magnetization over time
    ax3.clear()
    ax3.plot(time_history, mag_history_x, 'r-', label='Mx (transverse)', linewidth=2)
    ax3.plot(time_history, mag_history_y, 'b-', label='My (transverse)', linewidth=2)
    ax3.plot(time_history, mag_history_magnitude, 'k-', label='|M_xy| (magnitude)', linewidth=2.5)

    # Mark pulse times
    ax3.axvline(t1_end, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='90° pulse end')
    ax3.axvline(t2_end, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='180° pulse start')
    ax3.axvline(t3_end, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label='180° pulse end')
    ax3.axvline(t4_end, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Echo time')

    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Net Magnetization', fontsize=11)
    ax3.set_title('Sum of All Spins (Net Magnetization)', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, t4_end])
    ax3.set_ylim([-1.1, 1.1])

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    return []

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                    interval=50, blit=False)

# Save as GIF
print("Saving GIF...")
writer = PillowWriter(fps=15)
anim.save('hahn_echo_diffusion.gif', writer=writer)
print("Animation saved as 'hahn_echo_diffusion.gif'")

plt.close()
print("Done!")

