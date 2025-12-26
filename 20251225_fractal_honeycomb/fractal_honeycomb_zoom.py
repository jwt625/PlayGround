import math
import numpy as np
import cairosvg
import imageio.v3 as iio
from pathlib import Path

# ----------------------------
# Geometry: pointy-top hexagons
# ----------------------------
def hex_points(cx, cy, r):
    pts = []
    for i in range(6):
        ang = math.radians(60 * i - 30)  # pointy-top
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts

def hex_grid_centers(r, w, h, pad=2.0):
    """
    Create hex centers covering a box [-w/2, w/2] x [-h/2, h/2] with margin.
    Pointy-top hex tiling.
    """
    dx = math.sqrt(3) * r
    dy = 1.5 * r

    # cover with extra padding
    W = w * pad
    H = h * pad

    # approximate row/col counts
    n_rows = int(H / dy) + 6
    n_cols = int(W / dx) + 6

    centers = []
    y0 = -H / 2
    for row in range(n_rows):
        y = y0 + row * dy
        x0 = -W / 2 + (dx / 2 if row % 2 else 0.0)
        for col in range(n_cols):
            x = x0 + col * dx
            centers.append((x, y))
    return centers

def svg_polygon(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)

# ----------------------------
# SVG frame generator
# ----------------------------
def make_frame_svg(
    size_px=800,
    r0=10.0,
    k=3.0,
    t=0.0,
    background="#ffffff",
    stroke="#9a9a9a",
    stroke2="#6f6f6f",
):
    """
    One cycle: zoom-out by factor k while crossfading fine->coarse.
    t in [0,1).
    """

    # Zoom factor for this frame (zooming OUT means things get smaller on screen)
    zoom = k ** t
    s = 1.0 / zoom  # scale applied to world -> screen

    # Crossfade for perfect loop
    a_fine = 1.0 - t
    a_coarse = t

    # Keep stroke widths visually stable under scaling:
    # If we scale geometry by s, then stroke-width should be stroke_screen / s in world units.
    stroke_screen = 1.1
    sw_world = stroke_screen / s

    # Coarse layer: its own stroke a bit heavier (optional)
    stroke_screen2 = 1.4
    sw_world2 = stroke_screen2 / s

    # World canvas extents (in "world units")
    # We render centered at (0,0) and use a viewBox so everything is easy.
    view_w = size_px
    view_h = size_px

    # Hex centers for each layer (in world units)
    centers_fine = hex_grid_centers(r0, view_w, view_h, pad=2.2)
    centers_coarse = hex_grid_centers(r0 * k, view_w, view_h, pad=2.2)

    # SVG header: viewBox in world coordinates, then transform group scales by s
    # so the "camera" zooms out.
    svg = []
    svg.append(f"""<svg xmlns="http://www.w3.org/2000/svg"
        width="{size_px}" height="{size_px}"
        viewBox="{-(view_w/2):.1f} {-(view_h/2):.1f} {view_w:.1f} {view_h:.1f}">
        <rect x="{-(view_w/2):.1f}" y="{-(view_h/2):.1f}" width="{view_w:.1f}" height="{view_h:.1f}" fill="{background}"/>
    """)

    # Optional subtle vignette for depth (can be removed)
    svg.append(f"""
      <defs>
        <radialGradient id="vign" cx="50%" cy="45%" r="70%">
          <stop offset="0%" stop-color="#ffffff" stop-opacity="0.0"/>
          <stop offset="100%" stop-color="#000000" stop-opacity="0.06"/>
        </radialGradient>
      </defs>
      <rect x="{-(view_w/2):.1f}" y="{-(view_h/2):.1f}" width="{view_w:.1f}" height="{view_h:.1f}" fill="url(#vign)"/>
    """)

    # Global transform (camera)
    svg.append(f"""<g transform="scale({s:.6f})">""")

    # Fine layer (small cells)
    svg.append(f"""<g opacity="{a_fine:.6f}" fill="none" stroke="{stroke}" stroke-width="{sw_world:.6f}">""")
    for cx, cy in centers_fine:
        pts = hex_points(cx, cy, r0)
        svg.append(f"""<polygon points="{svg_polygon(pts)}" />""")
    svg.append("</g>")

    # Coarse layer (big cells)
    svg.append(f"""<g opacity="{a_coarse:.6f}" fill="none" stroke="{stroke2}" stroke-width="{sw_world2:.6f}">""")
    for cx, cy in centers_coarse:
        pts = hex_points(cx, cy, r0 * k)
        svg.append(f"""<polygon points="{svg_polygon(pts)}" />""")
    svg.append("</g>")

    svg.append("</g>")   # end camera group
    svg.append("</svg>")
    return "\n".join(svg)

# ----------------------------
# Render SVG -> PNG -> GIF
# ----------------------------
def main():
    out_dir = Path("honeycomb_frames")
    out_dir.mkdir(exist_ok=True)

    size_px = 900
    frames = 90           # 60–120 is typical for GIF
    fps = 30
    k = 3.0               # self-similarity ratio
    r0 = 11.0             # base cell size

    png_paths = []
    for i in range(frames):
        t = i / frames  # [0,1)
        svg = make_frame_svg(size_px=size_px, r0=r0, k=k, t=t)

        svg_path = out_dir / f"frame_{i:04d}.svg"
        png_path = out_dir / f"frame_{i:04d}.png"
        svg_path.write_text(svg, encoding="utf-8")

        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=size_px, output_height=size_px)
        png_paths.append(png_path)

    # Assemble GIF
    images = [iio.imread(p) for p in png_paths]
    iio.imwrite(
        "fractal_honeycomb_zoom.gif",
        images,
        duration=1.0 / fps,
        loop=0
    )
    print("Wrote fractal_honeycomb_zoom.gif")

if __name__ == "__main__":
    main()
