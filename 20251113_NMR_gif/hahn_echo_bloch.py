import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# Parameters
n_spins = 20  # Number of spins
n_frames = 250  # Total frames for animation
pulse_90_duration = 8  # Duration of 90° pulse in frames (faster)
pulse_180_duration = 10  # Duration of 180° pulse in frames (faster)
tau = 70  # Time between pulses (in frame units)

# Define time segments
# Segment 1: 90° pulse (0 to pulse_90_duration)
# Segment 2: Free precession (pulse_90_duration to pulse_90_duration + tau)
# Segment 3: 180° pulse (pulse_90_duration + tau to pulse_90_duration + tau + pulse_180_duration)
# Segment 4: Refocusing (pulse_90_duration + tau + pulse_180_duration to end)

t1_end = pulse_90_duration
t2_end = t1_end + tau
t3_end = t2_end + pulse_180_duration
t4_end = t3_end + tau

# Time points
t = np.linspace(0, t4_end, n_frames)

# Initialize spin positions (all start along +z after 90° pulse)
# Add some spread in frequencies to show dephasing
frequencies = np.linspace(-0.1, 0.1, n_spins)  # Different precession frequencies

def bloch_sphere_wireframe(ax):
    """Draw the Bloch sphere wireframe"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan', edgecolor='none')
    ax.plot_wireframe(x, y, z, alpha=0.2, color='gray', linewidth=0.5)

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

def hahn_echo_evolution(time_idx, frequencies):
    """
    Calculate spin positions during Hahn echo sequence
    Sequence: 90°x - tau - 180°y - tau - echo
    """
    time = t[time_idx]

    # Initialize spins along +z (equilibrium)
    x = np.zeros(n_spins)
    y = np.zeros(n_spins)
    z = np.ones(n_spins)

    # Segment 1: 90° pulse around x-axis
    if time <= t1_end:
        # Rotate from +z to +y
        angle = (time / pulse_90_duration) * (np.pi / 2)  # 0 to 90°
        x, y, z = rotate_x(x, y, z, angle)
        phase_label = f'90° Pulse (X-axis)\nAngle: {np.degrees(angle):.1f}°'

    # Segment 2: Free precession in xy-plane
    elif time <= t2_end:
        # Spins dephase in xy-plane
        # After 90° pulse around x, spins are along -y and precess in xy-plane
        time_since_pulse = time - t1_end
        phases = frequencies * time_since_pulse
        x = -np.sin(phases)  # Precess from -y
        y = -np.cos(phases)  # Start from -y
        z = np.zeros(n_spins)
        phase_label = f'Free Precession\nTime: {time_since_pulse:.1f}/{tau:.0f}'

    # Segment 3: 180° pulse around y-axis
    elif time <= t3_end:
        # Get positions just before 180° pulse
        time_before_180 = tau
        phases = frequencies * time_before_180
        x_before = -np.sin(phases)
        y_before = -np.cos(phases)
        z_before = np.zeros(n_spins)

        # Apply 180° rotation around y-axis
        pulse_progress = (time - t2_end) / pulse_180_duration
        angle = pulse_progress * np.pi  # 0 to 180°
        x, y, z = rotate_y(x_before, y_before, z_before, angle)
        phase_label = f'180° Pulse (Y-axis)\nAngle: {np.degrees(angle):.1f}°'

    # Segment 4: Refocusing - spins rephase
    else:
        # After 180° pulse, spins are inverted and continue precessing
        time_after_180 = time - t3_end
        # The 180° pulse inverts the phases, so they refocus
        phases = frequencies * (tau - time_after_180)
        x = -np.sin(phases)
        y = -np.cos(phases)
        z = np.zeros(n_spins)
        phase_label = f'Refocusing (Echo)\nTime: {time_after_180:.1f}/{tau:.0f}'

    return x, y, z, phase_label

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def init():
    """Initialize animation"""
    ax.clear()
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Hahn Echo Sequence - Bloch Sphere', fontsize=14, fontweight='bold')
    return []

def animate(frame):
    """Animation function"""
    ax.clear()

    # Draw Bloch sphere
    bloch_sphere_wireframe(ax)

    # Draw axes
    ax.plot([0, 1.3], [0, 0], [0, 0], 'k-', linewidth=2, alpha=0.6)
    ax.plot([0, 0], [0, 1.3], [0, 0], 'k-', linewidth=2, alpha=0.6)
    ax.plot([0, 0], [0, 0], [0, 1.3], 'k-', linewidth=2, alpha=0.6)
    ax.text(1.4, 0, 0, 'X', fontsize=12, fontweight='bold')
    ax.text(0, 1.4, 0, 'Y', fontsize=12, fontweight='bold')
    ax.text(0, 0, 1.4, 'Z', fontsize=12, fontweight='bold')

    # Get spin positions and phase label
    x, y, z, phase_text = hahn_echo_evolution(frame, frequencies)

    # Draw spins as arrows from origin
    for i in range(n_spins):
        ax.quiver(0, 0, 0, x[i], y[i], z[i],
                 color='red', alpha=0.7, arrow_length_ratio=0.15, linewidth=2)

    # Draw spin endpoints
    ax.scatter(x, y, z, c='red', s=50, alpha=0.8)

    # Add sequence information
    ax.text2D(0.05, 0.95, phase_text, transform=ax.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Hahn Echo Sequence - Bloch Sphere', fontsize=14, fontweight='bold')

    # Set viewing angle
    ax.view_init(elev=20, azim=frame * 360 / n_frames)  # Rotate view

    return []

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, 
                    interval=50, blit=False)

# Save as GIF
print("Saving GIF...")
writer = PillowWriter(fps=20)
anim.save('hahn_echo_bloch.gif', writer=writer)
print("Animation saved as 'hahn_echo_bloch.gif'")

plt.close()
print("Done!")

