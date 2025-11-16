import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set larger font sizes for mobile viewing
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})

# Define frequency range (normalized units, starting from 0)
omega = np.linspace(0, 3.0, 500)

# Waveguide parameters - ideal dispersion (linear)
n_0 = 1.0  # Linear refractive index (sets slope: beta = n*omega/c, here c=1)

# Pump parameters
omega_p = 1.5  # Pump frequency
P_pump = 1.0  # Pump power (normalized)
n2 = 0.1  # Nonlinear refractive index coefficient

# Calculate dispersion curves (ideal = straight lines from origin)
# beta = n * omega (assuming c = 1)
# Zero power: beta = n_0 * omega
beta_zero_power = n_0 * omega

# With pump on, effective refractive index changes:
# For pump (SPM): n_eff = n_0 + n2 * P_pump
# For signal/idler (XPM): n_eff = n_0 + 2 * n2 * P_pump

n_eff_pump = n_0 + n2 * P_pump
n_eff_signal_idler = n_0 + 2 * n2 * P_pump

# Dispersion curves with pump on (different slopes, all through origin)
beta_pump = n_eff_pump * omega
beta_signal_idler = n_eff_signal_idler * omega

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(10, 9))

ax.plot(omega, beta_zero_power, 'b-', linewidth=3, label='Zero Power', alpha=0.8)
ax.plot(omega, beta_signal_idler, 'r--', linewidth=3, label='Signal/Idler (Pump On)', alpha=0.8)
ax.plot(omega, beta_pump, 'g-.', linewidth=3, label='Pump (Pump On)', alpha=0.8)

# Mark pump frequency
ax.axvline(omega_p, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Pump Frequency')

ax.set_xlabel('Frequency ω (normalized)', fontweight='bold')
ax.set_ylabel('Propagation Constant β (normalized)', fontweight='bold')
ax.set_title('Nonlinear Waveguide Dispersion Curves\nwith Pump-Induced Phase Mismatch', 
             fontweight='bold', pad=20)
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.set_aspect('auto')

# Adjust layout
plt.tight_layout()

# Save as PNG
plt.savefig('dispersion_curve.png', dpi=150, bbox_inches='tight')
print("Saved PNG: dispersion_curve.png")

# Create interactive Plotly figure
fig_plotly = go.Figure()

fig_plotly.add_trace(go.Scatter(
    x=omega, y=beta_zero_power,
    mode='lines',
    name='Zero Power',
    line=dict(color='blue', width=3),
    hovertemplate='ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=omega, y=beta_signal_idler,
    mode='lines',
    name='Signal/Idler (Pump On)',
    line=dict(color='red', width=3, dash='dash'),
    hovertemplate='ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=omega, y=beta_pump,
    mode='lines',
    name='Pump (Pump On)',
    line=dict(color='green', width=3, dash='dot'),
    hovertemplate='ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

# Add vertical line for pump frequency
fig_plotly.add_vline(
    x=omega_p, 
    line_dash="dot", 
    line_color="gray",
    annotation_text="Pump Frequency",
    annotation_position="top"
)

fig_plotly.update_layout(
    title=dict(
        text='Nonlinear Waveguide Dispersion Curves<br>with Pump-Induced Phase Mismatch',
        font=dict(size=24, family='Arial Black')
    ),
    xaxis_title=dict(text='Frequency ω (normalized)', font=dict(size=20)),
    yaxis_title=dict(text='Propagation Constant β (normalized)', font=dict(size=20)),
    font=dict(size=16),
    hovermode='closest',
    showlegend=True,
    legend=dict(font=dict(size=16)),
    width=900,
    height=850,
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    plot_bgcolor='white'
)

# Save as HTML
fig_plotly.write_html('dispersion_curve.html')
print("Saved HTML: dispersion_curve.html")

print("\nPlot parameters:")
print(f"Pump frequency: {omega_p}")
print(f"Pump power: {P_pump}")
print(f"Linear refractive index n₀: {n_0}")
print(f"Nonlinear refractive index n₂: {n2}")
print(f"Effective index (pump, SPM): {n_eff_pump:.3f}")
print(f"Effective index (signal/idler, XPM): {n_eff_signal_idler:.3f}")
print("\nFiles saved successfully!")

