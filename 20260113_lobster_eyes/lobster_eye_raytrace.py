#!/usr/bin/env python3
"""
Lobster Eye X-ray Optics Ray Tracing Animation
Demonstrates wide FOV focusing onto a curved focal plane.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import io

# ============== GEOMETRY PARAMETERS ==============
NUM_CHANNELS = 17           # Number of channels in the lobster eye
CHANNEL_ANGULAR_SPAN = 100  # Total angular span of optic in degrees (-50 to +50)
CHANNEL_DEPTH = 0.45        # Depth (length) of each channel
OPTIC_RADIUS = 1.5          # Radius of the curved optic surface (center of curvature at origin)
FOCAL_RADIUS = 0.75         # Radius of the curved focal plane (half of optic radius)
NUM_RAYS = 150              # Number of rays per frame (6x denser)
MAX_REFLECTIONS = 6         # Maximum reflections per ray

# Animation parameters
ANGLE_RANGE = 60            # ±60 degrees sweep
NUM_FRAMES = 61             # Number of frames in animation
FPS = 10                    # Frames per second


def create_lobster_eye_channels(num_channels, channel_depth, optic_radius, angular_span_deg=100):
    """
    Create channel geometry: channels arranged on curved surface with straight walls.

    The lobster eye is a section of a sphere. Each channel:
    - Opens outward (away from center of curvature) to receive incoming light
    - Has straight walls that point radially toward the center of curvature
    - Center of curvature is at origin (0, 0)
    - Channels are on the UPPER part of the sphere (y > 0)
    - Channels are adjacent with no gaps (walls are shared between neighboring channels)
    """
    channels = []
    half_span = angular_span_deg / 2

    # Calculate angular width of each channel
    angular_width = angular_span_deg / num_channels

    # Generate channel edge angles (one more than num_channels for wall positions)
    edge_angles = np.linspace(-half_span, half_span, num_channels + 1) * np.pi / 180

    for i in range(num_channels):
        left_angle = edge_angles[i]
        right_angle = edge_angles[i + 1]
        center_angle = (left_angle + right_angle) / 2

        # Left wall - at left_angle, pointing radially inward
        lx1 = optic_radius * np.sin(left_angle)
        ly1 = optic_radius * np.cos(left_angle)
        # Radial direction at left edge
        lrx, lry = -np.sin(left_angle), -np.cos(left_angle)
        lx2 = lx1 + channel_depth * lrx
        ly2 = ly1 + channel_depth * lry

        # Right wall - at right_angle, pointing radially inward
        rx1 = optic_radius * np.sin(right_angle)
        ry1 = optic_radius * np.cos(right_angle)
        # Radial direction at right edge
        rrx, rry = -np.sin(right_angle), -np.cos(right_angle)
        rx2 = rx1 + channel_depth * rrx
        ry2 = ry1 + channel_depth * rry

        channels.append({
            'left_wall': ((lx1, ly1), (lx2, ly2)),
            'right_wall': ((rx1, ry1), (rx2, ry2)),
            'angle': center_angle
        })
    return channels


def line_intersection(p1, p2, p3, p4):
    """Find intersection of line segment p1-p2 with p3-p4. Returns (t, u) parameters."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None, None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    return t, u


def reflect_ray(direction, wall_start, wall_end):
    """Reflect ray direction off a wall."""
    wx = wall_end[0] - wall_start[0]
    wy = wall_end[1] - wall_start[1]
    wall_len = np.sqrt(wx**2 + wy**2)
    wx, wy = wx / wall_len, wy / wall_len
    
    # Normal to wall (perpendicular)
    nx, ny = -wy, wx
    
    # Reflect: r = d - 2(d·n)n
    dot = direction[0] * nx + direction[1] * ny
    rx = direction[0] - 2 * dot * nx
    ry = direction[1] - 2 * dot * ny
    
    return (rx, ry)


def trace_ray(start, direction, channels, max_reflections, optic_radius=1.5):
    """
    Trace a single ray through the optic.
    Returns (path, entry_index) where entry_index is the index where ray enters optic region.
    """
    path = [start]
    pos = start
    dir_vec = direction
    entry_index = None  # Index in path where ray enters the optic region

    for step in range(max_reflections + 1):
        best_t = float('inf')
        best_hit = None
        best_wall = None

        # Check intersection with all channel walls
        for ch in channels:
            for wall_name in ['left_wall', 'right_wall']:
                w = ch[wall_name]
                t, u = line_intersection(pos, (pos[0] + dir_vec[0] * 10, pos[1] + dir_vec[1] * 10),
                                         w[0], w[1])
                if t is not None and 0.001 < t < best_t and 0 <= u <= 1:
                    best_t = t
                    best_hit = (pos[0] + dir_vec[0] * t * 10, pos[1] + dir_vec[1] * t * 10)
                    best_wall = w

        if best_hit is None:
            # No wall hit - extend ray until it exits the view
            # Calculate how far to extend (to y = -1 or beyond view)
            if abs(dir_vec[1]) > 0.01:
                t_to_bottom = (-1.0 - pos[1]) / dir_vec[1]
                if t_to_bottom > 0:
                    end_pos = (pos[0] + dir_vec[0] * t_to_bottom, -1.0)
                else:
                    end_pos = (pos[0] + dir_vec[0] * 5, pos[1] + dir_vec[1] * 5)
            else:
                end_pos = (pos[0] + dir_vec[0] * 5, pos[1] + dir_vec[1] * 5)

            # Check if we're crossing into the optic region (crossing the optic arc)
            if entry_index is None:
                # Find where ray crosses the optic radius
                dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
                if dist_from_center > optic_radius:
                    # Ray is outside, find entry point
                    # Solve |pos + t*dir|^2 = optic_radius^2
                    a = dir_vec[0]**2 + dir_vec[1]**2
                    b = 2 * (pos[0] * dir_vec[0] + pos[1] * dir_vec[1])
                    c = pos[0]**2 + pos[1]**2 - optic_radius**2
                    disc = b**2 - 4*a*c
                    if disc >= 0:
                        t1 = (-b - np.sqrt(disc)) / (2*a)
                        t2 = (-b + np.sqrt(disc)) / (2*a)
                        t_entry = t1 if t1 > 0.001 else t2
                        if t_entry > 0.001:
                            entry_point = (pos[0] + dir_vec[0] * t_entry,
                                          pos[1] + dir_vec[1] * t_entry)
                            # Only add entry point if it's before the end
                            if entry_point[1] > end_pos[1]:
                                path.append(entry_point)
                                entry_index = len(path) - 1

            path.append(end_pos)
            break

        # Mark entry into optic region on first wall hit
        if entry_index is None:
            entry_index = len(path)

        path.append(best_hit)
        pos = best_hit
        dir_vec = reflect_ray(dir_vec, best_wall[0], best_wall[1])

    # If entry_index is still None, the ray never entered the optic region
    # Set it to 1 (after the start point) so everything after start is "transmitted"
    if entry_index is None:
        entry_index = 1

    return path, entry_index


def create_frame(angle_deg, channels, focal_radius, optic_radius):
    """Create a single frame of the animation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 3.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    ax.set_title(f'Lobster Eye Ray Tracing - Incident Angle: {angle_deg:.1f}°',
                 color='white', fontsize=14)
    ax.axis('off')

    # Draw curved focal plane - centered at origin, below the optic
    focal_angles = np.linspace(-80, 80, 100) * np.pi / 180
    focal_x = focal_radius * np.sin(focal_angles)
    focal_y = focal_radius * np.cos(focal_angles)  # Upper half of circle at origin
    ax.plot(focal_x, focal_y, 'c-', linewidth=2.5, alpha=0.8, label='Focal plane')

    # Draw optic curve (reference) - where channels are mounted
    optic_angles = np.linspace(-55, 55, 100) * np.pi / 180
    optic_x = optic_radius * np.sin(optic_angles)
    optic_y = optic_radius * np.cos(optic_angles)
    ax.plot(optic_x, optic_y, 'white', linewidth=1, alpha=0.3, linestyle='--')

    # Draw channel walls
    for ch in channels:
        for wall_name in ['left_wall', 'right_wall']:
            w = ch[wall_name]
            ax.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'gold', linewidth=1.5)

    # Ray direction: angle=0 means coming straight down, positive angle = from the right
    angle_rad = angle_deg * np.pi / 180
    dir_x = np.sin(angle_rad)
    dir_y = -np.cos(angle_rad)

    # Generate parallel rays - start from far away in the direction opposite to propagation
    perp_x, perp_y = np.cos(angle_rad), np.sin(angle_rad)  # Perpendicular to ray direction
    ray_starts = []
    spread = 2.5  # Width of the ray bundle

    for i in range(NUM_RAYS):
        offset = (i / (NUM_RAYS - 1) - 0.5) * spread * 2
        # Start rays far from the optic, coming toward it
        start_x = offset * perp_x - dir_x * 4
        start_y = offset * perp_y - dir_y * 4
        ray_starts.append((start_x, start_y))

    # Trace and draw rays
    for start in ray_starts:
        path, entry_idx = trace_ray(start, (dir_x, dir_y), channels, MAX_REFLECTIONS, optic_radius)

        # Draw ray path
        if len(path) >= 2:
            # Draw path segments
            for i in range(len(path) - 1):
                # Green for incident (before entering optic), red for after
                if i < entry_idx:
                    color = '#00ff88'
                    alpha = 0.7
                    linewidth = 0.8
                else:
                    color = '#ff6b6b'
                    alpha = 0.5
                    linewidth = 0.6
                ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]],
                       color=color, linewidth=linewidth, alpha=alpha)

            # Arrow on incident ray (green part)
            if entry_idx > 0 and entry_idx < len(path):
                ax.annotate('', xy=path[entry_idx], xytext=path[0],
                           arrowprops=dict(arrowstyle='->', color='#00ff88', lw=0.8))

    # Convert figure to image - use fixed bounds, no bbox_inches='tight'
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                facecolor=fig.get_facecolor(), pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def main():
    """Generate the animation."""
    print("Creating lobster eye geometry...")
    channels = create_lobster_eye_channels(NUM_CHANNELS, CHANNEL_DEPTH,
                                           OPTIC_RADIUS, CHANNEL_ANGULAR_SPAN)

    print(f"Generating {NUM_FRAMES} frames...")
    frames = []
    angles = np.linspace(-ANGLE_RANGE, ANGLE_RANGE, NUM_FRAMES)

    for i, angle in enumerate(angles):
        print(f"  Frame {i+1}/{NUM_FRAMES} (angle={angle:.1f}°)")
        frame = create_frame(angle, channels, FOCAL_RADIUS, OPTIC_RADIUS)
        frames.append(frame)

    # Save as GIF
    output_path = "lobster_eye_raytrace.gif"
    print(f"Saving animation to {output_path}...")
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=1000//FPS, loop=0)
    print("Done!")


if __name__ == "__main__":
    main()

