import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Set larger font sizes for mobile viewing
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12
})

# Define frequency range (normalized units, starting from 0)
omega = np.linspace(0, 3.0, 500)

# Waveguide parameters - ideal dispersion (linear)
n_0 = 1.0  # Linear refractive index

# FWM process parameters
omega_s = 1.0   # Signal frequency
omega_i = 2.0   # Idler frequency
omega_p = 1.5   # Pump frequency
n2 = 0.1        # Nonlinear refractive index coefficient

# Animation parameters
P_max = 2.0     # Maximum pump power (2x the original)
n_frames = 60   # Number of frames

# Calculate dispersion curves (zero power baseline)
beta_zero_power = n_0 * omega

# Create figure
fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)

# Initialize plot elements
line_zero, = ax.plot(omega, beta_zero_power, 'b-', linewidth=3, label='Zero Power', alpha=0.8)
line_si, = ax.plot([], [], 'r--', linewidth=3, label='Signal/Idler (Pump On)', alpha=0.8)
line_pump, = ax.plot([], [], 'g-.', linewidth=3, label='Pump (Pump On)', alpha=0.8)

# FWM points - Pump OFF (static)
point_s_off, = ax.plot(omega_s, n_0 * omega_s, 'bo', markersize=12, label=f'Signal (ω={omega_s})', zorder=5)
point_i_off, = ax.plot(omega_i, n_0 * omega_i, 'bs', markersize=12, label=f'Idler (ω={omega_i})', zorder=5)
point_p_off, = ax.plot(omega_p, n_0 * omega_p, 'b^', markersize=12, label=f'Pump (ω={omega_p})', zorder=5)

# FWM points - Pump ON (dynamic)
point_s_on, = ax.plot([], [], 'ro', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)
point_i_on, = ax.plot([], [], 'rs', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)
point_p_on, = ax.plot([], [], 'g^', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)

ax.set_xlabel('Frequency ω (normalized)', fontweight='bold')
ax.set_ylabel('Propagation Constant β (normalized)', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.set_aspect('auto')
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(-0.2, 3.5)

# Add vertical dashed lines for frequencies
ax.axvline(omega_s, color='red', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
ax.axvline(omega_i, color='red', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
ax.axvline(omega_p, color='green', linestyle=':', linewidth=2, alpha=0.6, zorder=1)

# Calculate maximum values for inset sizing
n_eff_pump_max = n_0 + n2 * P_max
n_eff_signal_idler_max = n_0 + 2 * n2 * P_max
beta_s_max = n_eff_signal_idler_max * omega_s
beta_i_max = n_eff_signal_idler_max * omega_i
beta_p_max = n_eff_pump_max * omega_p

# Inset elements (will be updated)
y_start = 0.05
x_pump = 0.15
x_si = 0.5
x_mismatch = 0.85

# Calculate scale to fit maximum values
max_height = max(2 * beta_p_max, beta_s_max + beta_i_max)
scale = 0.85 / max_height  # Use 0.85 to leave some margin

# Create inset for phase matching diagram
ax_inset = fig.add_axes([0.55, 0.15, 0.38, 0.35])
ax_inset.set_facecolor('white')
ax_inset.set_xlim(0, 1.2)
ax_inset.set_ylim(0, 1.0)
ax_inset.axis('off')

# Initialize inset arrows and text (will be cleared and redrawn each frame)
inset_artists = []

# Add power text annotation
power_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, 
                     fontsize=18, fontweight='bold', 
                     ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

def init():
    line_si.set_data([], [])
    line_pump.set_data([], [])
    point_s_on.set_data([], [])
    point_i_on.set_data([], [])
    point_p_on.set_data([], [])
    return line_si, line_pump, point_s_on, point_i_on, point_p_on

def update(frame):
    global inset_artists
    
    # Calculate current pump power
    P_pump = P_max * frame / (n_frames - 1)
    
    # Calculate effective indices
    n_eff_pump = n_0 + n2 * P_pump
    n_eff_signal_idler = n_0 + 2 * n2 * P_pump
    
    # Update dispersion curves
    beta_pump = n_eff_pump * omega
    beta_signal_idler = n_eff_signal_idler * omega
    
    line_si.set_data(omega, beta_signal_idler)
    line_pump.set_data(omega, beta_pump)
    
    # Update FWM points
    beta_s_on = n_eff_signal_idler * omega_s
    beta_i_on = n_eff_signal_idler * omega_i
    beta_p_on = n_eff_pump * omega_p
    
    point_s_on.set_data([omega_s], [beta_s_on])
    point_i_on.set_data([omega_i], [beta_i_on])
    point_p_on.set_data([omega_p], [beta_p_on])
    
    # Calculate phase mismatch
    delta_beta_on = beta_s_on + beta_i_on - 2 * beta_p_on
    
    # Update power text
    power_text.set_text(f'P = {P_pump:.2f}')
    
    # Clear previous inset artists
    for artist in inset_artists:
        artist.remove()
    inset_artists = []
    
    # Redraw inset
    # First pump arrow
    arr1 = ax_inset.arrow(x_pump, y_start, 0, beta_p_on * scale, 
                          head_width=0.05, head_length=0.03, fc='green', ec='green', linewidth=3)
    txt1 = ax_inset.text(x_pump - 0.05, y_start + beta_p_on * scale / 2, 'β_p', 
                         fontsize=16, fontweight='bold', color='green', ha='right', va='center')
    
    # Second pump arrow
    y_second = y_start + beta_p_on * scale
    arr2 = ax_inset.arrow(x_pump, y_second, 0, beta_p_on * scale, 
                          head_width=0.05, head_length=0.03, fc='green', ec='green', linewidth=3)
    txt2 = ax_inset.text(x_pump - 0.05, y_second + beta_p_on * scale / 2, 'β_p', 
                         fontsize=16, fontweight='bold', color='green', ha='right', va='center')
    
    y_2bp_top = y_start + 2 * beta_p_on * scale
    
    # Signal arrow
    arr3 = ax_inset.arrow(x_si, y_start, 0, beta_s_on * scale,
                          head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=3)
    txt3 = ax_inset.text(x_si + 0.08, y_start + beta_s_on * scale / 2, 'β_s',
                         fontsize=16, fontweight='bold', color='red', ha='left', va='center')

    # Idler arrow
    y_idler = y_start + beta_s_on * scale
    arr4 = ax_inset.arrow(x_si, y_idler, 0, beta_i_on * scale,
                          head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=3)
    txt4 = ax_inset.text(x_si + 0.08, y_idler + beta_i_on * scale / 2, 'β_i',
                         fontsize=16, fontweight='bold', color='red', ha='left', va='center')

    y_si_top = y_start + (beta_s_on + beta_i_on) * scale

    # Horizontal dashed lines
    line1, = ax_inset.plot([x_pump + 0.08, x_si - 0.08], [y_2bp_top, y_2bp_top],
                           'k--', linewidth=1.5, alpha=0.5)
    line2, = ax_inset.plot([x_pump + 0.08, x_si - 0.08], [y_si_top, y_si_top],
                           'k--', linewidth=1.5, alpha=0.5)

    # Phase mismatch arrow
    from matplotlib.patches import FancyArrowPatch
    arr5 = FancyArrowPatch((x_mismatch, y_2bp_top), (x_mismatch, y_si_top),
                           arrowstyle='<->', color='black', lw=3, mutation_scale=20)
    ax_inset.add_patch(arr5)
    txt5 = ax_inset.text(x_mismatch + 0.08, (y_si_top + y_2bp_top) / 2,
                         f'Δβ={delta_beta_on:.2f}',
                         fontsize=16, fontweight='bold', color='black', va='center')

    # Baseline
    line3, = ax_inset.plot([0, 1.2], [y_start, y_start], 'k-', linewidth=2)

    # Store all artists
    inset_artists = [arr1, txt1, arr2, txt2, arr3, txt3, arr4, txt4,
                     line1, line2, arr5, txt5, line3]

    return [line_si, line_pump, point_s_on, point_i_on, point_p_on, power_text] + inset_artists

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                     interval=100, blit=False, repeat=True)

# Save as GIF
print("Saving GIF (this may take a minute)...")
writer = PillowWriter(fps=10)
anim.save('dispersion_animation.gif', writer=writer, dpi=100)

print(f"\nAnimation saved successfully!")
print(f"Number of frames: {n_frames}")
print(f"Pump power range: 0 to {P_max}")
print(f"File: dispersion_animation.gif")

