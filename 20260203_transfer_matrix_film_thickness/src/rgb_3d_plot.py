"""
Interactive 3D RGB scatter plot of wafer pixels.
Outputs an HTML file you can open in a browser to explore the color distribution.
"""
import cv2
import numpy as np
import plotly.graph_objects as go
from ellipse_cache import get_ellipse_params, create_ellipse_mask
from paths import IMAGE_PATH, OUTPUT_ANALYSIS


def rgb_to_hex(r, g, b):
    """Convert RGB to hex color string."""
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def create_rgb_3d_plot(image_path, n_samples=10000):
    """
    Create interactive 3D scatter plot of RGB values.

    Parameters:
        image_path: Path to image
        n_samples: Number of pixels to sample (for performance)
    """
    # Load cached ellipse params
    params = get_ellipse_params(image_path)

    # Load image
    print("Loading image...")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create ellipse mask
    print("Creating ellipse mask...")
    mask = create_ellipse_mask(img.shape, params)

    # Get pixels inside wafer
    wafer_pixels = img_rgb[mask == 1]
    n_total = len(wafer_pixels)
    print(f"Total wafer pixels: {n_total:,}")

    # Random sample for visualization
    print(f"Sampling {n_samples:,} pixels for visualization...")
    idx = np.random.choice(n_total, min(n_samples, n_total), replace=False)
    sampled = wafer_pixels[idx]

    r = sampled[:, 0]
    g = sampled[:, 1]
    b = sampled[:, 2]

    # Create color strings for each point
    colors = [rgb_to_hex(r[i], g[i], b[i]) for i in range(len(r))]

    # Create 3D scatter plot
    print("Creating 3D plot...")
    fig = go.Figure(data=[go.Scatter3d(
        x=r,
        y=g,
        z=b,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.7
        ),
        hovertemplate='R: %{x}<br>G: %{y}<br>B: %{z}<extra></extra>'
    )])

    # Add reference point for pure white (glare)
    fig.add_trace(go.Scatter3d(
        x=[255],
        y=[255],
        z=[255],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='x'),
        text=['White (glare)'],
        textposition='top center',
        name='Reference: White'
    ))

    # Add reference for pure blue
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[255],
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='diamond'),
        text=['Pure Blue'],
        textposition='top center',
        name='Reference: Blue'
    ))

    fig.update_layout(
        title=f'RGB Distribution of Wafer Pixels ({n_samples:,} samples)',
        scene=dict(
            xaxis_title='Red',
            yaxis_title='Green',
            zaxis_title='Blue',
            xaxis=dict(range=[0, 255]),
            yaxis=dict(range=[0, 255]),
            zaxis=dict(range=[0, 255]),
        ),
        width=1000,
        height=800,
    )

    # Save as HTML
    output_path = str(OUTPUT_ANALYSIS / 'rgb_3d_plot.html')
    fig.write_html(output_path)
    print(f"\nInteractive 3D plot saved to: {output_path}")
    print("Open this file in your browser to explore the RGB distribution.")

    return fig


if __name__ == '__main__':
    create_rgb_3d_plot(str(IMAGE_PATH), n_samples=15000)
