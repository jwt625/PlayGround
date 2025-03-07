

#%%
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interact

# Define a list of prime numbers to choose from for frequencies.
prime_options = [101, 103, 107, 109, 113, 127, 131, 137, 139]

def plot_lissajous(delta=np.pi/2, a=101, b=103):
    # Generate time values over one full period.
    t = np.linspace(0, 2 * np.pi, 10000)
    
    # Compute x and y coordinates using the Lissajous formulas.
    x = 1.6*np.sin(a * t + delta)
    y = np.sin(b * t)
    
    # Create the interactive plot using Plotly.
    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(
        title=f"Lissajous Curve (a={a}, b={b}, δ={delta:.2f})",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )
    
    fig.show()

# Use ipywidgets to create interactive controls.
interact(
    plot_lissajous,
    delta=widgets.FloatSlider(
        min=0, max=2*np.pi, step=0.1, value=np.pi/2, description="Phase Shift"
    ),
    a=widgets.Dropdown(
        options=prime_options, value=101, description="Frequency a"
    ),
    b=widgets.Dropdown(
        options=prime_options, value=103, description="Frequency b"
    )
)

# %%
import numpy as np
import plotly.graph_objects as go
import imageio
import os
import shutil

def generate_frames():
    # Create (or recreate) a directory to store temporary frame images.
    frames_dir = "frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    # Lissajous curve parameters
    a = 101        # Frequency a (prime)
    b = 103        # Frequency b (prime)
    delta = np.pi/2  # Fixed phase shift

    num_frames = 60  # Total number of frames in the animation
    t_full = np.linspace(0, 2 * np.pi, 10000)  # Full parameter range for the curve

    file_names = []

    # Generate each frame by progressively increasing the portion of the curve drawn.
    for i in range(num_frames):
        # Determine the maximum t value for this frame.
        t_max = (i + 1) / num_frames * (2 * np.pi)
        # Select only those t values up to t_max.
        mask = t_full <= t_max
        t_current = t_full[mask]

        # Compute the x and y coordinates for the current segment.
        x = 1.6*np.sin(a * t_current + delta)
        y = np.sin(b * t_current)

        # Create a Plotly figure for this frame.
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
        fig.update_layout(
            title=f"Lissajous Curve Drawing Animation (Frame {i+1}/{num_frames})",
            xaxis_title="x",
            yaxis_title="y",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            template="plotly_white"
        )

        # Save the current frame as a PNG file.
        file_name = os.path.join(frames_dir, f"frame_{i:03d}.png")
        fig.write_image(file_name)
        file_names.append(file_name)

    # Read the generated images and create an animated GIF.
    images = []
    for file_name in file_names:
        images.append(imageio.imread(file_name))
    gif_filename = "lissajous_drawing.gif"
    imageio.mimsave(gif_filename, images, duration=0.1)
    print(f"Animated GIF saved as {gif_filename}")

# if __name__ == '__main__':
generate_frames()

# %%
