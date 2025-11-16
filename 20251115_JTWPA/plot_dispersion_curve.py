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

# FWM process parameters
omega_s = 1.0   # Signal frequency
omega_i = 2.0   # Idler frequency
omega_p = 1.5   # Pump frequency (degenerate: 2*omega_p = omega_s + omega_i)
P_pump = 1.0    # Pump power (normalized)
n2 = 0.1        # Nonlinear refractive index coefficient

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

# Calculate beta values at specific frequencies for FWM
# Pump off (zero power)
beta_s_off = n_0 * omega_s
beta_i_off = n_0 * omega_i
beta_p_off = n_0 * omega_p

# Pump on
beta_s_on = n_eff_signal_idler * omega_s  # Signal experiences XPM
beta_i_on = n_eff_signal_idler * omega_i  # Idler experiences XPM
beta_p_on = n_eff_pump * omega_p          # Pump experiences SPM

# Phase matching condition: Δβ = β_s + β_i - 2*β_p
delta_beta_off = beta_s_off + beta_i_off - 2 * beta_p_off
delta_beta_on = beta_s_on + beta_i_on - 2 * beta_p_on

# Create matplotlib figure with inset
fig = plt.figure(figsize=(10, 9))
ax = plt.subplot(111)

ax.plot(omega, beta_zero_power, 'b-', linewidth=3, label='Zero Power', alpha=0.8)
ax.plot(omega, beta_signal_idler, 'r--', linewidth=3, label='Signal/Idler (Pump On)', alpha=0.8)
ax.plot(omega, beta_pump, 'g-.', linewidth=3, label='Pump (Pump On)', alpha=0.8)

# Mark the FWM points - Pump OFF
ax.plot(omega_s, beta_s_off, 'bo', markersize=12, label=f'Signal (ω={omega_s})', zorder=5)
ax.plot(omega_i, beta_i_off, 'bs', markersize=12, label=f'Idler (ω={omega_i})', zorder=5)
ax.plot(omega_p, beta_p_off, 'b^', markersize=12, label=f'Pump (ω={omega_p})', zorder=5)

# Mark the FWM points - Pump ON
ax.plot(omega_s, beta_s_on, 'ro', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)
ax.plot(omega_i, beta_i_on, 'rs', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)
ax.plot(omega_p, beta_p_on, 'g^', markersize=12, markerfacecolor='none', markeredgewidth=3, zorder=5)

ax.set_xlabel('Frequency ω (normalized)', fontweight='bold')
ax.set_ylabel('Propagation Constant β (normalized)', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9, fontsize=12)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.set_aspect('auto')
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(-0.2, 3.5)

# Create inset for phase matching diagram (pump ON case)
ax_inset = fig.add_axes([0.55, 0.15, 0.38, 0.35])  # [left, bottom, width, height]
ax_inset.set_facecolor('white')
ax_inset.set_xlim(0, 1.2)
ax_inset.set_ylim(0, 1.0)
ax_inset.axis('off')

# Scale factor to fit arrows in inset
scale = 0.25

# Draw 2 pump arrows (connected vertically)
y_start = 0.1
arrow_width = 0.08
x_pump = 0.15

# First pump arrow
ax_inset.arrow(x_pump, y_start, 0, beta_p_on * scale,
               head_width=0.05, head_length=0.03, fc='green', ec='green', linewidth=3)
ax_inset.text(x_pump - 0.05, y_start + beta_p_on * scale / 2, 'β_p',
              fontsize=16, fontweight='bold', color='green', ha='right', va='center')

# Second pump arrow (stacked on top)
y_second = y_start + beta_p_on * scale
ax_inset.arrow(x_pump, y_second, 0, beta_p_on * scale,
               head_width=0.05, head_length=0.03, fc='green', ec='green', linewidth=3)
ax_inset.text(x_pump - 0.05, y_second + beta_p_on * scale / 2, 'β_p',
              fontsize=16, fontweight='bold', color='green', ha='right', va='center')

# Total 2*beta_p height
y_2bp_top = y_start + 2 * beta_p_on * scale

# Draw signal + idler arrows (connected vertically, next to pump)
x_si = 0.5

# Signal arrow
ax_inset.arrow(x_si, y_start, 0, beta_s_on * scale,
               head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=3)
ax_inset.text(x_si + 0.08, y_start + beta_s_on * scale / 2, 'β_s',
              fontsize=16, fontweight='bold', color='red', ha='left', va='center')

# Idler arrow (stacked on top of signal)
y_idler = y_start + beta_s_on * scale
ax_inset.arrow(x_si, y_idler, 0, beta_i_on * scale,
               head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=3)
ax_inset.text(x_si + 0.08, y_idler + beta_i_on * scale / 2, 'β_i',
              fontsize=16, fontweight='bold', color='red', ha='left', va='center')

# Total beta_s + beta_i height
y_si_top = y_start + (beta_s_on + beta_i_on) * scale

# Draw horizontal dashed lines to show the difference
ax_inset.plot([x_pump + 0.08, x_si - 0.08], [y_2bp_top, y_2bp_top],
              'k--', linewidth=1.5, alpha=0.5)
ax_inset.plot([x_pump + 0.08, x_si - 0.08], [y_si_top, y_si_top],
              'k--', linewidth=1.5, alpha=0.5)

# Draw phase mismatch arrow
x_mismatch = 0.85
ax_inset.annotate('', xy=(x_mismatch, y_si_top), xytext=(x_mismatch, y_2bp_top),
                  arrowprops=dict(arrowstyle='<->', color='black', lw=3))
ax_inset.text(x_mismatch + 0.08, (y_si_top + y_2bp_top) / 2,
              f'Δβ={delta_beta_on:.2f}',
              fontsize=16, fontweight='bold', color='black', va='center')

# Add baseline
ax_inset.plot([0, 1.2], [y_start, y_start], 'k-', linewidth=2)

# Adjust layout
plt.tight_layout()

# Save as PNG
plt.savefig('dispersion_curve.png', dpi=150, bbox_inches='tight')
print("Saved PNG: dispersion_curve.png")

# Create interactive Plotly figure
fig_plotly = go.Figure()

# Dispersion curves
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

# FWM points - Pump OFF
fig_plotly.add_trace(go.Scatter(
    x=[omega_s], y=[beta_s_off],
    mode='markers',
    name=f'Signal (ω={omega_s}, off)',
    marker=dict(color='blue', size=12, symbol='circle'),
    hovertemplate='Signal OFF<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=[omega_i], y=[beta_i_off],
    mode='markers',
    name=f'Idler (ω={omega_i}, off)',
    marker=dict(color='blue', size=12, symbol='square'),
    hovertemplate='Idler OFF<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=[omega_p], y=[beta_p_off],
    mode='markers',
    name=f'Pump (ω={omega_p}, off)',
    marker=dict(color='blue', size=12, symbol='triangle-up'),
    hovertemplate='Pump OFF<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

# FWM points - Pump ON
fig_plotly.add_trace(go.Scatter(
    x=[omega_s], y=[beta_s_on],
    mode='markers',
    name=f'Signal (ω={omega_s}, on)',
    marker=dict(color='red', size=12, symbol='circle-open', line=dict(width=3)),
    hovertemplate='Signal ON<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=[omega_i], y=[beta_i_on],
    mode='markers',
    name=f'Idler (ω={omega_i}, on)',
    marker=dict(color='red', size=12, symbol='square-open', line=dict(width=3)),
    hovertemplate='Idler ON<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

fig_plotly.add_trace(go.Scatter(
    x=[omega_p], y=[beta_p_on],
    mode='markers',
    name=f'Pump (ω={omega_p}, on)',
    marker=dict(color='green', size=12, symbol='triangle-up-open', line=dict(width=3)),
    hovertemplate='Pump ON<br>ω: %{x:.3f}<br>β: %{y:.3f}<extra></extra>'
))

# Add annotations for phase mismatch
fig_plotly.add_annotation(
    x=omega_p + 0.3, y=(2*beta_p_off + beta_s_off + beta_i_off)/2,
    text=f'Δβ={delta_beta_off:.3f}<br>(matched)',
    showarrow=False,
    font=dict(size=14, color='blue'),
    bgcolor='rgba(255,255,255,0.8)'
)

fig_plotly.add_annotation(
    x=omega_p - 0.3, y=(2*beta_p_on + beta_s_on + beta_i_on)/2,
    text=f'Δβ={delta_beta_on:.3f}<br>(mismatch)',
    showarrow=False,
    font=dict(size=14, color='red'),
    bgcolor='rgba(255,255,255,0.8)'
)

fig_plotly.update_layout(
    xaxis_title=dict(text='Frequency ω (normalized)', font=dict(size=20)),
    yaxis_title=dict(text='Propagation Constant β (normalized)', font=dict(size=20)),
    font=dict(size=16),
    hovermode='closest',
    showlegend=True,
    legend=dict(font=dict(size=14)),
    width=900,
    height=850,
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-0.1, 3.1]),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[-0.2, 3.5]),
    plot_bgcolor='white'
)

# Save as HTML
fig_plotly.write_html('dispersion_curve.html')
print("Saved HTML: dispersion_curve.html")

print("\n" + "="*60)
print("FWM PROCESS PARAMETERS")
print("="*60)
print(f"Signal frequency ω_s: {omega_s}")
print(f"Idler frequency ω_i:  {omega_i}")
print(f"Pump frequency ω_p:   {omega_p}")
print(f"Frequency matching:   2ω_p = {2*omega_p} = ω_s + ω_i = {omega_s + omega_i}")
print(f"\nPump power: {P_pump}")
print(f"Linear refractive index n₀: {n_0}")
print(f"Nonlinear refractive index n₂: {n2}")
print(f"\nEffective index (pump, SPM): {n_eff_pump:.3f}")
print(f"Effective index (signal/idler, XPM): {n_eff_signal_idler:.3f}")
print("\n" + "="*60)
print("PHASE MATCHING")
print("="*60)
print(f"Pump OFF:  Δβ = β_s + β_i - 2β_p = {delta_beta_off:.6f} (matched!)")
print(f"Pump ON:   Δβ = β_s + β_i - 2β_p = {delta_beta_on:.6f} (mismatch)")
print(f"Mismatch induced by pump: {delta_beta_on - delta_beta_off:.6f}")
print("="*60)
print("\nFiles saved successfully!")

