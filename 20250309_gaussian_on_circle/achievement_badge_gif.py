#!/usr/bin/env python3
"""
Generate an achievement badge GIF with "CERTIFIED OUTSIDE FIVE SIGMA" banner
around a rotating 3D Gaussian distribution wrapped on a circle.
"""

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import imageio.v2 as imageio
import io
from PIL import Image, ImageDraw, ImageFont
import math

def add_achievement_banner(image_array, text="CERTIFIED OUTSIDE FIVE SIGMA"):
    """Add a circular achievement banner around the image"""
    # Convert numpy array to PIL Image
    img = Image.fromarray(image_array)
    width, height = img.size
    
    # Create a new image with some padding for the banner
    banner_width = 140  # smaller banner to make central plot appear larger
    new_size = max(width, height) + 2 * banner_width
    new_img = Image.new('RGBA', (new_size, new_size), (255, 255, 255, 0))
    
    # Calculate center position
    center_x = center_y = new_size // 2
    
    # Paste the original image in the center
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    # Create drawing context
    draw = ImageDraw.Draw(new_img)
    
    # Draw outer circle (gold/yellow border)
    outer_radius = new_size // 2 - 10
    inner_radius = outer_radius - banner_width
    
    # Draw the banner ring with gradient-like effect (dark blue like Ghost in the Shell)
    dark_blue = (0, 51, 102)  # Dark blue color like the laughing man icon
    for i in range(10):
        radius = outer_radius - i * 2
        alpha = 255 - i * 15
        color = (*dark_blue, alpha)  # Dark blue with varying alpha
        draw.ellipse([center_x - radius, center_y - radius,
                     center_x + radius, center_y + radius],
                    outline=color, width=3)

    # Draw inner circle border (same thickness as outer circle)
    for i in range(10):
        radius = inner_radius + i * 2
        alpha = 255 - i * 15
        color = (*dark_blue, alpha)
        draw.ellipse([center_x - radius, center_y - radius,
                     center_x + radius, center_y + radius],
                    outline=color, width=3)
    
    # Add text around the circle
    font_size = 72  # 3x larger than before (was 24)
    font = None

    # Try multiple font paths for different systems
    font_paths = [
        "/System/Library/Fonts/Arial Bold.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",   # macOS fallback
        "Arial-Bold",                             # System font name
        "arial.ttf",                              # Windows
        "DejaVuSans-Bold.ttf"                     # Linux
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Using font: {font_path}")
            break
        except:
            continue

    # If no TrueType font found, create a larger default font
    if font is None:
        print("Warning: Using default font - text may be smaller")
        font = ImageFont.load_default()

    # Calculate text positioning around the circle
    text_radius = (outer_radius + inner_radius) // 2

    # Add extra space between end and beginning by adding padding characters
    # This creates a gap between "SIGMA" and "CERTIFIED"
    spaced_text = text.replace(" ", "  ") + "      "  # Extra spaces at the end
    text_chars = list(spaced_text)
    num_chars = len(text_chars)

    for i, char in enumerate(text_chars):
        if char == ' ':
            continue

        # Calculate angle for this character - start from top, go clockwise
        angle = (i / num_chars) * 2 * math.pi - math.pi / 2  # Start from top

        # Calculate position
        char_x = center_x + text_radius * math.cos(angle)
        char_y = center_y + text_radius * math.sin(angle)

        # Calculate rotation angle for the character
        # Rotate in the opposite direction (clockwise) and remove the -90° offset
        char_angle = -math.degrees(angle)  # Negative for clockwise rotation, no extra offset

        # Create a temporary image for the rotated character (larger for bigger font)
        char_img = Image.new('RGBA', (150, 150), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)

        # Draw character in the center of temp image
        bbox = char_draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        char_draw.text((75 - char_width//2, 75 - char_height//2), char,
                      fill=(*dark_blue, 255), font=font)  # Dark blue text

        # Rotate the character
        rotated_char = char_img.rotate(char_angle, expand=True)

        # Paste the rotated character
        paste_x = int(char_x - rotated_char.width // 2)
        paste_y = int(char_y - rotated_char.height // 2)
        new_img.paste(rotated_char, (paste_x, paste_y), rotated_char)
    

    
    # Convert back to numpy array
    return np.array(new_img.convert('RGB'))

def create_gaussian_figure():
    """Create the 3D Gaussian figure with all traces"""
    # --- Parameters ---
    sigma = 0.2  # sigma equals the circle radius
    x = np.linspace(-10*sigma, 10*sigma, 1000)
    x_extend = np.logspace(np.log10(10*sigma), np.log10(1000*sigma), 100)
    x = np.concatenate((-np.flip(x_extend), x, x_extend))
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)
    
    # --- Stereographic Projection ---
    # Mapping from the real line to the unit circle:
    # u(x) = 2x/(1+x^2) and v(x) = (x^2 - 1)/(1+x^2)
    u = 2*x/(1+x**2)
    v = (x**2 - 1)/(1+x**2)
    
    # Build a mesh that fills between the top (curve) and its projection (z=0)
    n = len(u)
    x_vertices = np.concatenate([u, u])
    y_vertices = np.concatenate([v, v])
    z_vertices = np.concatenate([pdf, np.zeros_like(pdf)])
    intensity = np.concatenate([pdf, pdf])
    
    # Build faces (two triangles per segment)
    faces_i = []
    faces_j = []
    faces_k = []
    for i_pt in range(n - 1):
        faces_i.append(i_pt)
        faces_j.append(i_pt + 1)
        faces_k.append(i_pt + n)
        
        faces_i.append(i_pt + 1)
        faces_j.append(i_pt + 1 + n)
        faces_k.append(i_pt + n)

    mesh_trace = go.Mesh3d(
        x=x_vertices,
        y=y_vertices,
        z=z_vertices,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        intensity=intensity,
        colorscale='Viridis',
        opacity=0.5,
        showscale=False,
        name='Shaded Surface'
    )

    curve_trace_line = go.Scatter3d(
        x=u,
        y=v,
        z=pdf,
        mode='lines',
        line=dict(width=2, color='darkblue'),
        name='Gaussian Curve Line'
    )

    curve_trace_markers = go.Scatter3d(
        x=u,
        y=v,
        z=pdf,
        mode='markers',
        marker=dict(
            size=4,
            color=pdf,
            colorscale='Viridis'
        ),
        name='Gaussian Curve Markers'
    )

    proj_trace = go.Scatter3d(
        x=u,
        y=v,
        z=np.zeros_like(pdf),
        mode='lines',
        line=dict(dash='dash', width=2, color='black'),
        name='Projection (z=0)'
    )

    fig_plotly = go.Figure(data=[mesh_trace, curve_trace_line, curve_trace_markers, proj_trace])

    # Add annotation with arrow at the top of the circle (point (0,1,0))
    fig_plotly.update_layout(
        width=900, height=900, margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        scene=dict(
            xaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
            yaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
            zaxis=dict(title='', showticklabels=False, ticks='', showgrid=True),
            annotations=[
                dict(
                    x=0,
                    y=1,
                    z=0,
                    text="you are here",
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=4,
                    arrowcolor="red",
                    ax=0,   # horizontal offset (in pixels)
                    ay=-120, # vertical offset (in pixels) - adjusted for larger size
                    font=dict(color="black", size=30)  # larger font for bigger image
                )
            ]
        )
    )
    
    return fig_plotly

def generate_preview():
    """Generate a preview of the first frame as PNG"""
    print("Creating 3D Gaussian figure...")
    fig_plotly = create_gaussian_figure()

    print("Generating preview frame...")
    # Set camera to first position (angle = 0)
    angle = 0
    d = 1.8        # distance of the camera from the center
    cam_z = 0.3    # fixed z coordinate for the camera

    angle_rad = np.deg2rad(angle)
    cam_x = d * np.cos(angle_rad)
    cam_y = d * np.sin(angle_rad)

    # Update the camera view for the 3D scene.
    fig_plotly.update_layout(
        scene_camera=dict(
            eye=dict(x=cam_x, y=cam_y, z=cam_z)
        )
    )

    # Export the current view as a PNG image (requires kaleido).
    img_bytes = pio.to_image(fig_plotly, format='png', width=900, height=900)
    image = imageio.imread(io.BytesIO(img_bytes))

    # Add the achievement banner
    enhanced_image = add_achievement_banner(image)

    # Save as PNG
    preview_filename = 'achievement_preview.png'
    imageio.imwrite(preview_filename, enhanced_image)
    print(f"Preview saved as {preview_filename}")

def main():
    """Generate the achievement badge GIF"""
    print("Creating 3D Gaussian figure...")
    fig_plotly = create_gaussian_figure()

    print("Generating achievement badge GIF...")
    frames = []
    num_frames = 72  # number of frames in the orbit shot
    d = 1.8        # distance of the camera from the center
    cam_z = 0.3    # fixed z coordinate for the camera

    for angle in np.linspace(0, 360, num_frames, endpoint=False):
        print(f"Processing frame {len(frames)+1}/{num_frames} (angle: {angle:.1f}°)")
        angle_rad = np.deg2rad(angle)
        cam_x = d * np.cos(angle_rad)
        cam_y = d * np.sin(angle_rad)

        # Update the camera view for the 3D scene.
        fig_plotly.update_layout(
            scene_camera=dict(
                eye=dict(x=cam_x, y=cam_y, z=cam_z)
            )
        )

        # Export the current view as a PNG image (requires kaleido).
        img_bytes = pio.to_image(fig_plotly, format='png', width=900, height=900)  # 1.5x larger
        image = imageio.imread(io.BytesIO(img_bytes))

        # Add the achievement banner
        enhanced_image = add_achievement_banner(image)
        frames.append(enhanced_image)

    gif_filename = 'orbit_plotly_achievement.gif'
    imageio.mimsave(gif_filename, frames, fps=20)
    print(f"Achievement badge GIF saved as {gif_filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        generate_preview()
    else:
        main()
