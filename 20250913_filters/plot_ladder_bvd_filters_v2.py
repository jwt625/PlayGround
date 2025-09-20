#!/usr/bin/env python3
"""
Plot ladder filter designs with mechanical resonances using BVD model.

Focus on S-parameters (S11, S21) with linear frequency scale near the passband.
Uses fixed Q = 1000 and electromechanical coupling k² = 10%.

Ladder filter topology:
- N parallel resonances (shunt to ground)  
- N+1 series resonances (in signal path)
- N = 2, 3, 4 configurations

BVD model parameters:
- Series resonance: fs = 1/(2π√(LmCm))
- Parallel resonance: fp = fs√(1 + Cm/C0) = fs√(1 + k²/(1-k²))
- Quality factor: Q = 2πfsLm/Rm
- Coupling factor: k² = Cm/(Cm + C0)
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def bvd_impedance(f, fs, k_squared, Q, Z0=50.0):
    """
    Calculate BVD resonator impedance using correct model with proper scaling.

    Parameters:
    f: frequency array [Hz]
    fs: series resonance frequency [Hz]
    k_squared: electromechanical coupling factor k² = Cm/(Cm + C0)
    Q: quality factor
    Z0: characteristic impedance for scaling (default 50Ω)

    Returns:
    Z: complex impedance

    BVD Model:
    - Series resonance: fs = 1/(2π√(Lm*Cm))
    - Parallel resonance: fp = fs * √(1 + Cm/C0)
    - Coupling factor: k² = Cm/(Cm + C0)
    - Quality factor: Q = ωs*Lm/Rm where Rm is motional resistance

    Component values are scaled to give reasonable impedance levels around Z0.
    """
    omega = 2 * np.pi * f
    omega_s = 2 * np.pi * fs

    # From k² = Cm/(Cm + C0), we get Cm/C0 = k²/(1 - k²)
    Cm_C0_ratio = k_squared / (1 - k_squared)

    # Scale Cm to give reasonable impedance levels
    # For a ladder filter, we want each series element to contribute only a small
    # fraction of the total impedance. With N series elements, target Rm ≈ Z0/(5*N)
    # But since we don't know N here, use a conservative scaling
    # Choose Rm ≈ Z0/20 for low series resistance
    Rm_target = Z0 / 10.0
    Cm = 1.0 / (omega_s * Q * Rm_target)

    # From fs = 1/(2π√(Lm*Cm)), we get Lm
    Lm = 1.0 / (omega_s**2 * Cm)

    # Motional resistance from Q
    Rm = omega_s * Lm / Q

    # From Cm/C0 ratio, we get C0
    C0 = Cm / Cm_C0_ratio

    # BVD impedance calculation
    # Motional branch: Rm + jωLm + 1/(jωCm)
    Z_motional = Rm + 1j*omega*Lm + 1.0/(1j*omega*Cm)

    # Static capacitance: 1/(jωC0)
    Z_static = 1.0/(1j*omega*C0)

    # Total impedance: Z_static in parallel with Z_motional
    Z_total = 1.0 / (1.0/Z_static + 1.0/Z_motional)

    return Z_total

def series_abcd(Z):
    """ABCD matrix for series impedance Z."""
    return np.array([[1, Z], [0, 1]])

def shunt_abcd(Y):
    """ABCD matrix for shunt admittance Y."""
    return np.array([[1, 0], [Y, 1]])

def abcd_to_s(abcd, Z0=50):
    """
    Convert ABCD matrix to S-parameters.

    Parameters:
    abcd: 2x2 ABCD matrix (can be array of matrices for frequency sweep)
    Z0: characteristic impedance

    Returns:
    S11, S21: S-parameters
    """
    if abcd.ndim == 3:  # Array of matrices
        S11 = np.zeros(abcd.shape[0], dtype=complex)
        S21 = np.zeros(abcd.shape[0], dtype=complex)

        for i in range(abcd.shape[0]):
            A, B, C, D = abcd[i, 0, 0], abcd[i, 0, 1], abcd[i, 1, 0], abcd[i, 1, 1]
            denom = A + B/Z0 + C*Z0 + D
            S11[i] = (A + B/Z0 - C*Z0 - D) / denom
            S21[i] = 2 / denom

        return S11, S21
    else:  # Single matrix
        A, B, C, D = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        denom = A + B/Z0 + C*Z0 + D
        S11 = (A + B/Z0 - C*Z0 - D) / denom
        S21 = 2 / denom
        return S11, S21

def ladder_filter_sparameters(f, N_parallel, fs_center=1e9, k_squared=0.1, Q=1000, Z0=50):
    """
    Calculate S-parameters for ladder filter using ABCD matrices.

    Parameters:
    f: frequency array [Hz]
    N_parallel: number of parallel resonances
    fs_center: center frequency [Hz]
    k_squared: coupling factor
    Q: quality factor
    Z0: characteristic impedance [ohms]

    Returns:
    S11, S21: reflection and transmission coefficients
    """
    N_series = N_parallel + 1

    # Generate resonance frequencies for testing (no spread)
    # All series resonators identical, all parallel resonators identical
    # But different between the two types

    # For series resonators: all have fs at passband center
    # This gives low impedance at fs_center
    fs_series = np.full(N_series, fs_center)

    # For parallel resonators: all have fp at passband center
    # Calculate fs needed so that fp = fs * sqrt(1 + k²/(1-k²)) = fs_center
    fs_for_parallel = fs_center / np.sqrt(1 + k_squared / (1 - k_squared))
    fs_parallel = np.full(N_parallel, fs_for_parallel)
    # print fs_series and fs_parallel
    print(f'fs_series = {fs_series}')
    print(f'fs_parallel = {fs_parallel}')

    # Debug: Check impedance levels at center frequency
    f_center_single = np.array([fs_center])
    Z_series_center = bvd_impedance(f_center_single, fs_center, k_squared, Q)
    Z_parallel_center = bvd_impedance(f_center_single, fs_for_parallel, k_squared, Q)

    print(f'\nAt center frequency ({fs_center/1e9:.3f} GHz):')
    print(f'Series resonator impedance: {Z_series_center[0]:.2f} Ω')
    print(f'Parallel resonator impedance: {Z_parallel_center[0]:.2f} Ω')
    print(f'Parallel resonator admittance: {1/Z_parallel_center[0]:.6f} S')

    # Calculate component values for reference
    omega_s = 2 * np.pi * fs_center
    Z0 = 50.0
    Rm_target = Z0 / 10.0
    Cm = 1.0 / (omega_s * Q * Rm_target)
    C0 = Cm * (1 - k_squared) / k_squared
    Lm = 1.0 / (omega_s**2 * Cm)
    print(f'Component values:')
    print(f'  Cm: {Cm*1e12:.2f} pF')
    print(f'  C0: {C0*1e12:.2f} pF')
    print(f'  Lm: {Lm*1e3:.2f} mH')
    print(f'  Rm: {Rm_target:.2f} Ω')
    print(f'Total series resistance for {N_series} elements: {N_series * Rm_target:.1f} Ω')

    # Initialize overall ABCD matrix for each frequency
    nfreq = len(f)
    ABCD_total = np.zeros((nfreq, 2, 2), dtype=complex)

    # Initialize as identity matrix
    for i in range(nfreq):
        ABCD_total[i] = np.eye(2)

    # Build ladder network using ABCD matrices
    # Topology: Series → Shunt → Series → Shunt → ... → Series
    # For N_parallel = 2: Series → Shunt → Series → Shunt → Series (3 series, 2 shunt)

    # Build the complete ladder
    for i in range(N_series + N_parallel):
        if i % 2 == 0:  # Even positions: Series elements
            series_idx = i // 2
            Z_series = bvd_impedance(f, fs_series[series_idx], k_squared, Q)
            for freq_idx in range(nfreq):
                ABCD_series = series_abcd(Z_series[freq_idx])
                ABCD_total[freq_idx] = ABCD_total[freq_idx] @ ABCD_series
        else:  # Odd positions: Shunt elements
            shunt_idx = i // 2
            Z_shunt = bvd_impedance(f, fs_parallel[shunt_idx], k_squared, Q)
            Y_shunt = 1.0 / Z_shunt
            for freq_idx in range(nfreq):
                ABCD_shunt = shunt_abcd(Y_shunt[freq_idx])
                ABCD_total[freq_idx] = ABCD_total[freq_idx] @ ABCD_shunt

    # Convert ABCD to S-parameters
    S11, S21 = abcd_to_s(ABCD_total, Z0)

    return S11, S21

def plot_ladder_filters_interactive():
    """Plot interactive ladder filter S-parameters and impedances using Plotly."""

    # Frequency range around 1 GHz - wider range
    fs_center = 1e9  # 1 GHz
    f = np.linspace(0.6e9, 1.4e9, 10000)  # ±40% around center

    # Fixed parameters
    k_squared = 0.15  # 10% coupling
    Q = 200
    # Filter configurations
    N_values = [3, 4, 5]
    colors = ['blue', 'red', 'green']

    # Create subplots using plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'BVD Resonator Impedance Characteristics',
            'Transmission Coefficient S21',
            'Reflection Coefficient S11',
            'S-Parameters for N_par = 3 Configuration'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.update_layout(
        title_text=f'Ladder Filter with BVD Resonators (Q = {Q}, k² = {k_squared*100}%)',
        title_x=0.5,
        height=800,
        showlegend=True
    )

    # Plot 1: BVD impedance showing series and parallel resonance behavior
    f_bvd = np.linspace(0.8e9, 1.5e9, 10000)

    # Calculate series and parallel resonance frequencies
    fp = fs_center * np.sqrt(1 + k_squared / (1 - k_squared))

    # Plot impedance for series-type resonator (used in series branch)
    Z_series = bvd_impedance(f_bvd, fs_center, k_squared, Q)
    Z_series_db = 20 * np.log10(np.abs(Z_series))

    # Plot impedance for parallel-type resonator (used in shunt branch)
    fs_center_parallel = fs_center / np.sqrt(1 + k_squared / (1 - k_squared))
    Z_parallel = bvd_impedance(f_bvd, fs_center_parallel, k_squared, Q)
    Z_parallel_db = 20 * np.log10(np.abs(Z_parallel))

    # Combined impedance
    Z_toPlot = 1.0/Z_parallel
    Z_toPlot_db = 20 * np.log10(np.abs(Z_toPlot))

    print(f'fs_center_parallel = {fs_center_parallel/1e9:.3f} GHz')
    print(f'fs_center = {fs_center/1e9:.3f} GHz')

    # Add traces to subplot 1 (row=1, col=1)
    # fig.add_trace(
    #     go.Scatter(x=f_bvd/1e6, y=np.imag(Z_series), mode='lines', name='Series Resonator |Z|',
    #               line=dict(color='red', width=2), legendgroup='plot1'), row=1, col=1)

    # fig.add_trace(
    #     go.Scatter(x=f_bvd/1e6, y=np.imag(Z_parallel), mode='lines', name='Parallel Resonator |Z|',
    #               line=dict(color='blue', width=2), legendgroup='plot1'),        row=1, col=1
    # )

    # fig.add_trace(
    #     go.Scatter(x=f_bvd/1e6, y=np.imag(1.0/Z_parallel), mode='lines', name='Series + 1/Parallel |Z|',
    #               line=dict(color='green', width=2, dash='dash'), legendgroup='plot1'),        row=1, col=1
    # )

    fig.add_trace(
        go.Scatter(x=f_bvd/1e9, y=(Z_series_db), mode='lines', name='Series Resonator |Z|',
                  line=dict(color='red', width=2), legendgroup='plot1'),        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=f_bvd/1e9, y=(Z_parallel_db), mode='lines', name='Parallel Resonator |Z|',
                  line=dict(color='blue', width=2), legendgroup='plot1'),        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=f_bvd/1e9, y=(Z_toPlot_db), mode='lines', name='Series + 1/Parallel |Z|',
                  line=dict(color='green', width=2, dash='dash'), legendgroup='plot1'),        row=1, col=1
    )

    # Add vertical lines for resonance frequencies
    fig.add_vline(x=fs_center/1e9, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text=f'fs = {fs_center/1e9:.3f} GHz', row=1, col=1)
    fig.add_vline(x=fp/1e9, line_dash="dash", line_color="blue", opacity=0.7,
                  annotation_text=f'fp = {fp/1e9:.3f} GHz', row=1, col=1)

    # Update subplot 1 axes
    fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=1)
    fig.update_yaxes(title_text="|Z| (dB-Ω)", range=[0, 180], row=1, col=1)
    # fig.update_yaxes(title_text="|Z| (dB-Ω)", range=[-100, 100], row=1, col=1)

    # Plot 2: S21 (Transmission)
    for i, N in enumerate(N_values):
        S11, S21 = ladder_filter_sparameters(f, N, fs_center, k_squared, Q)
        S21_db = 20 * np.log10(np.abs(S21))
        fig.add_trace(
            go.Scatter(x=f/1e9, y=S21_db, mode='lines',
                      name=f'N_par = {N}, N_ser = {N+1}',
                      line=dict(color=colors[i], width=2), legendgroup='plot2'),
            row=1, col=2
        )

    # Add horizontal line for -3dB
    fig.add_hline(y=-3, line_dash="dot", line_color="black", opacity=0.7,
                  annotation_text='-3dB', row=1, col=2)

    # Update subplot 2 axes
    fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=2)
    fig.update_yaxes(title_text="S21 (dB)", range=[-120, 5], row=1, col=2)

    # Plot 3: S11 (Reflection)
    for i, N in enumerate(N_values):
        S11, S21 = ladder_filter_sparameters(f, N, fs_center, k_squared, Q)
        S11_db = 20 * np.log10(np.abs(S11))
        fig.add_trace(
            go.Scatter(x=f/1e9, y=S11_db, mode='lines',
                      name=f'N_par = {N}, N_ser = {N+1}',
                      line=dict(color=colors[i], width=2), legendgroup='plot3'),
            row=2, col=1
        )

    # Add horizontal line for -10dB
    fig.add_hline(y=-10, line_dash="dot", line_color="black", opacity=0.7,
                  annotation_text='-10dB', row=2, col=1)

    # Update subplot 3 axes
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
    fig.update_yaxes(title_text="S11 (dB)", range=[-120, 5], row=2, col=1)

    # Plot 4: Combined S-parameters for N=3
    N_demo = 3
    S11, S21 = ladder_filter_sparameters(f, N_demo, fs_center, k_squared, Q)
    S11_db = 20 * np.log10(np.abs(S11))
    S21_db = 20 * np.log10(np.abs(S21))

    fig.add_trace(
        go.Scatter(x=f/1e9, y=S21_db, mode='lines', name='S21 (Transmission)',
                  line=dict(color='blue', width=2), legendgroup='plot4'),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(x=f/1e9, y=S11_db, mode='lines', name='S11 (Reflection)',
                  line=dict(color='red', width=2), legendgroup='plot4'),
        row=2, col=2
    )

    # Add horizontal lines
    fig.add_hline(y=-3, line_dash="dot", line_color="blue", opacity=0.7,
                  annotation_text='-3dB', row=2, col=2)
    fig.add_hline(y=-10, line_dash="dot", line_color="red", opacity=0.7,
                  annotation_text='-10dB', row=2, col=2)

    # Update subplot 4 axes
    fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=2)
    fig.update_yaxes(title_text="S-Parameters (dB)", range=[-120, 5], row=2, col=2)

    # Show the interactive plot
    fig.show()

    # Save as HTML for interactivity
    fig.write_html("ladder_bvd_sparameters_interactive.html")

    # Also save static image for compatibility
    fig.write_image("ladder_bvd_sparameters.png", width=1500, height=800)


def plot_ladder_filters():
    """Plot ladder filter S-parameters and impedances using matplotlib (original version)."""

    # Frequency range around 1 GHz - wider range
    fs_center = 1e9  # 1 GHz
    f = np.linspace(0.9e9, 1.1e9, 1000)  # ±10% around center

    # Fixed parameters
    k_squared = 0.01  # 10% coupling
    Q = 1000

    # Filter configurations
    N_values = [2, 3, 4]
    colors = ['blue', 'red', 'green']

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Ladder Filter with BVD Resonators (Q = {Q}, k² = {k_squared*100}%)',
                 fontsize=16, fontweight='bold')

    # Plot 1: BVD impedance showing series and parallel resonance behavior
    ax1.set_title('BVD Resonator Impedance Characteristics')
    f_bvd = np.linspace(0.8e9, 1.5e9, 3000)

    # Calculate series and parallel resonance frequencies
    fp = fs_center * np.sqrt(1 + k_squared / (1 - k_squared))

    # Plot impedance for series-type resonator (used in series branch)
    Z_series = bvd_impedance(f_bvd, fs_center, k_squared, Q)
    Z_series_db = 20 * np.log10(np.abs(Z_series))
    ax1.plot(f_bvd/1e6, Z_series_db, 'red', linewidth=2, label='Series Resonator |Z|')

    # Plot impedance for parallel-type resonator (used in shunt branch)
    # For shunt, we care about admittance, so plot -|Z| to show low impedance at parallel resonance
    fs_center_parallel = fs_center / np.sqrt(1 + k_squared / (1 - k_squared))
    Z_parallel = bvd_impedance(f_bvd, fs_center_parallel, k_squared, Q)
    Z_parallel_db = 20 * np.log10(np.abs(Z_parallel))
    ax1.plot(f_bvd/1e6, Z_parallel_db, 'blue', linewidth=2, label='Parallel Resonator |Z|')
    # print fs_center_parallel and fs_center
    print(f'fs_center_parallel = {fs_center_parallel/1e6:.3f} MHz')
    print(f'fs_center = {fs_center/1e6:.3f} MHz')
    # plot Y_parallel + Z_series
    Z_toPlot = Z_series + 1.0/Z_parallel
    Z_toPlot_db = 20 * np.log10(np.abs(Z_toPlot))
    ax1.plot(f_bvd/1e6, Z_toPlot_db, 'green', linewidth=2, label='Series + 1/Parallel |Z|', linestyle='--')


    # Mark resonance frequencies
    ax1.axvline(x=fs_center/1e6, color='red', linestyle='--', alpha=0.7,
                label=f'fs = {fs_center/1e6:.3f} MHz')
    ax1.axvline(x=fp/1e6, color='blue', linestyle='--', alpha=0.7,
                label=f'fp = {fp/1e6:.3f} MHz')

    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('|Z| (dB-Ω)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 180])

    # Plot 2: S21 (Transmission)
    ax2.set_title('Transmission Coefficient S21')
    for i, N in enumerate(N_values):
        S11, S21 = ladder_filter_sparameters(f, N, fs_center, k_squared, Q)
        S21_db = 20 * np.log10(np.abs(S21))
        ax2.plot(f/1e6, S21_db, color=colors[i], linewidth=2,
                label=f'N_par = {N}, N_ser = {N+1}')

    ax2.axhline(y=-3, color='black', linestyle=':', alpha=0.7, label='-3dB')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('S21 (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-120, 5])

    # Plot 3: S11 (Reflection)
    ax3.set_title('Reflection Coefficient S11')
    for i, N in enumerate(N_values):
        S11, S21 = ladder_filter_sparameters(f, N, fs_center, k_squared, Q)
        S11_db = 20 * np.log10(np.abs(S11))
        ax3.plot(f/1e6, S11_db, color=colors[i], linewidth=2,
                label=f'N_par = {N}, N_ser = {N+1}')

    ax3.axhline(y=-10, color='black', linestyle=':', alpha=0.7, label='-10dB')
    ax3.set_xlabel('Frequency (MHz)')
    ax3.set_ylabel('S11 (dB)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([-120, 5])

    # Plot 4: Combined S-parameters for N=3
    ax4.set_title('S-Parameters for N_par = 3 Configuration')
    N_demo = 3
    S11, S21 = ladder_filter_sparameters(f, N_demo, fs_center, k_squared, Q)
    S11_db = 20 * np.log10(np.abs(S11))
    S21_db = 20 * np.log10(np.abs(S21))

    ax4.plot(f/1e6, S21_db, 'blue', linewidth=2, label='S21 (Transmission)')
    ax4.plot(f/1e6, S11_db, 'red', linewidth=2, label='S11 (Reflection)')
    ax4.axhline(y=-3, color='blue', linestyle=':', alpha=0.7, label='-3dB')
    ax4.axhline(y=-10, color='red', linestyle=':', alpha=0.7, label='-10dB')

    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('S-Parameters (dB)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([-120, 5])

    plt.tight_layout()
    plt.savefig('ladder_bvd_sparameters.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_filter_summary():
    """Print summary of ladder filter characteristics."""
    
    print("Ladder Filter with BVD Resonators - Summary")
    print("=" * 50)
    print("Fixed Parameters:")
    print("• Quality factor Q = 1000")
    print("• Electromechanical coupling k² = 10%")
    print("• Center frequency = 1 MHz")
    print("• Characteristic impedance Z₀ = 50Ω")
    print()
    
    print("Filter Configurations:")
    print("-" * 25)
    for N in [2, 3, 4]:
        print(f"• N = {N}: {N} parallel + {N+1} series = {2*N+1} total resonators")
    print()
    
    print("Key Features:")
    print("-" * 15)
    print("• Linear frequency scale focused on passband")
    print("• S11 and S21 parameters shown")
    print("• BVD model with series/parallel resonances")
    print("• Higher N gives sharper filter response")
    print("• Trade-off between selectivity and complexity")

if __name__ == "__main__":
    print("Plotting Ladder Filter with BVD Mechanical Resonators")
    print("Focus: S-parameters with linear frequency scale")
    print("=" * 60)

    print_filter_summary()
    print()

    # Generate interactive plots
    print("Generating interactive plots...")
    plot_ladder_filters_interactive()

    print("Plots saved as:")
    print("• ladder_bvd_sparameters_interactive.html (interactive)")
    print("• ladder_bvd_sparameters.png (static)")
    print()
    print("Open the HTML file in your browser for interactive plots!")
