#!/usr/bin/env python3
"""Debug script to understand the lobster eye geometry and ray tracing."""

import numpy as np
import matplotlib.pyplot as plt

# Import from main script
from lobster_eye_raytrace import (
    create_lobster_eye_channels, trace_ray,
    NUM_CHANNELS, CHANNEL_DEPTH, OPTIC_RADIUS, FOCAL_RADIUS,
    CHANNEL_ANGULAR_SPAN, MAX_REFLECTIONS
)

def debug_geometry():
    """Print and visualize the geometry."""
    channels = create_lobster_eye_channels(NUM_CHANNELS, CHANNEL_DEPTH,
                                           OPTIC_RADIUS, CHANNEL_ANGULAR_SPAN)

    print("=" * 60)
    print("LOBSTER EYE GEOMETRY DEBUG")
    print("=" * 60)
    print(f"Number of channels: {NUM_CHANNELS}")
    print(f"Angular span: {CHANNEL_ANGULAR_SPAN}°")
    print(f"Channel depth: {CHANNEL_DEPTH}")
    print(f"Optic radius: {OPTIC_RADIUS}")
    print(f"Focal radius: {FOCAL_RADIUS}")
    print()

    # Print first, middle, last channel info
    for idx in [0, NUM_CHANNELS // 2, NUM_CHANNELS - 1]:
        ch = channels[idx]
        print(f"Channel {idx} (angle={np.degrees(ch['angle']):.1f}°):")
        lw = ch['left_wall']
        rw = ch['right_wall']
        print(f"  Left wall:  ({lw[0][0]:.3f}, {lw[0][1]:.3f}) -> ({lw[1][0]:.3f}, {lw[1][1]:.3f})")
        print(f"  Right wall: ({rw[0][0]:.3f}, {rw[0][1]:.3f}) -> ({rw[1][0]:.3f}, {rw[1][1]:.3f})")
        print()

    # Create geometry visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Focal plane - centered at origin
    focal_angles = np.linspace(-80, 80, 100) * np.pi / 180
    focal_x = FOCAL_RADIUS * np.sin(focal_angles)
    focal_y = FOCAL_RADIUS * np.cos(focal_angles)

    # Optic curve
    optic_angles = np.linspace(-55, 55, 100) * np.pi / 180
    optic_x = OPTIC_RADIUS * np.sin(optic_angles)
    optic_y = OPTIC_RADIUS * np.cos(optic_angles)

    # Plot 1: Just the geometry
    ax = axes[0]
    ax.set_title("Geometry Only\n(Center of curvature at origin)")
    ax.set_aspect('equal')

    for ch in channels:
        for wall_name in ['left_wall', 'right_wall']:
            w = ch[wall_name]
            ax.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'b-', linewidth=2)

    ax.plot(focal_x, focal_y, 'r-', linewidth=2, label='Focal plane')
    ax.plot(optic_x, optic_y, 'g--', linewidth=1, alpha=0.5, label='Optic curve')
    ax.plot(0, 0, 'ko', markersize=5, label='Center')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Plot 2: Rays at 0 degrees (coming straight down)
    ax = axes[1]
    ax.set_title("Rays at 0° incidence\n(straight down)")
    ax.set_aspect('equal')

    for ch in channels:
        for wall_name in ['left_wall', 'right_wall']:
            w = ch[wall_name]
            ax.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'b-', linewidth=1)
    ax.plot(focal_x, focal_y, 'r-', linewidth=2)

    angle_rad = 0
    dir_x, dir_y = np.sin(angle_rad), -np.cos(angle_rad)
    print(f"Ray direction at 0°: ({dir_x:.3f}, {dir_y:.3f})")

    # Detailed trace for rays
    from lobster_eye_raytrace import reflect_ray, line_intersection

    # Test multiple ray positions
    for test_x in [-0.5, -0.15, 0.0, 0.05, 0.5]:
        print(f"\n--- Detailed trace for ray at x={test_x} ---")
        test_pos = (test_x, 3.0)
        test_dir = (0, -1)
        for step in range(3):
            print(f"Step {step}: pos={test_pos}, dir={test_dir}")
            best_t = float('inf')
            best_hit = None
            best_wall = None
            best_wall_name = None
            for ch in channels:
                for wall_name in ['left_wall', 'right_wall']:
                    w = ch[wall_name]
                    t, u = line_intersection(test_pos, (test_pos[0] + test_dir[0] * 10, test_pos[1] + test_dir[1] * 10), w[0], w[1])
                    if t is not None and 0.001 < t < best_t and 0 <= u <= 1:
                        best_t = t
                        best_hit = (test_pos[0] + test_dir[0] * t * 10, test_pos[1] + test_dir[1] * t * 10)
                        best_wall = w
                        best_wall_name = f"angle={np.degrees(ch['angle']):.1f} {wall_name}"
            if best_hit:
                print(f"  Hit {best_wall_name} at {best_hit}")
                print(f"  Wall: {best_wall}")
                new_dir = reflect_ray(test_dir, best_wall[0], best_wall[1])
                print(f"  New direction: {new_dir}")
                test_pos = best_hit
                test_dir = new_dir
            else:
                print("  No hit")
                break

    for i in range(20):
        start_x = (i - 10) * 0.15
        start_y = 3.0
        path, _ = trace_ray((start_x, start_y), (dir_x, dir_y), channels, MAX_REFLECTIONS)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'g-', linewidth=0.8, alpha=0.7)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(True, alpha=0.3)

    # Plot 3: Rays at 30 degrees
    ax = axes[2]
    ax.set_title("Rays at 30° incidence")
    ax.set_aspect('equal')

    for ch in channels:
        for wall_name in ['left_wall', 'right_wall']:
            w = ch[wall_name]
            ax.plot([w[0][0], w[1][0]], [w[0][1], w[1][1]], 'b-', linewidth=1)
    ax.plot(focal_x, focal_y, 'r-', linewidth=2)

    angle_rad = 30 * np.pi / 180
    dir_x, dir_y = np.sin(angle_rad), -np.cos(angle_rad)
    perp_x, perp_y = np.cos(angle_rad), np.sin(angle_rad)
    print(f"Ray direction at 30°: ({dir_x:.3f}, {dir_y:.3f})")

    for i in range(20):
        offset = (i - 10) * 0.2
        start_x = offset * perp_x - dir_x * 4
        start_y = offset * perp_y - dir_y * 4
        path, _ = trace_ray((start_x, start_y), (dir_x, dir_y), channels, MAX_REFLECTIONS)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'g-', linewidth=0.8, alpha=0.7)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('debug_geometry.png', dpi=150)
    print("\nSaved debug_geometry.png")
    plt.close()


def save_single_frames():
    """Save single frames for inspection."""
    from lobster_eye_raytrace import create_frame

    channels = create_lobster_eye_channels(NUM_CHANNELS, CHANNEL_DEPTH,
                                           OPTIC_RADIUS, CHANNEL_ANGULAR_SPAN)

    for angle in [0, 30, -30]:
        img = create_frame(angle, channels, FOCAL_RADIUS, OPTIC_RADIUS)
        filename = f'frame_angle_{angle}.png'
        img.save(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    debug_geometry()
    save_single_frames()

