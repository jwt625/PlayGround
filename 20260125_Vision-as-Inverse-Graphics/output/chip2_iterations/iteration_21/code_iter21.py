import gdsfactory as gf
import numpy as np

# Claude's Iteration 21: Fresh analysis of chip2.png
# Key observations from my visual inspection:
# 1. 3 ring resonators arranged horizontally at bottom
# 2. Electrode arcs (not full circles) around each ring - appear to be partial arcs
# 3. Grating couplers on left edge (horizontal line patterns)
# 4. Metal contact pads at top right
# 5. Horizontal bus waveguide coupling to rings
# 6. Metal routing traces from pads to electrodes

c = gf.Component("chip2_iter21_claude")

# === Parameters based on visual analysis ===
ring_radius = 10.0        # µm
ring_width = 0.5          # µm
coupling_gap = 0.20       # µm (iter9 found this optimal)
ring_spacing = 25.0       # µm (rings appear well-spaced in image)
bus_width = 0.5           # µm
num_rings = 3

# Electrode arc parameters - appear to be partial arcs, not full rings
electrode_inner_radius = 11.0  # µm
electrode_outer_radius = 15.0  # µm
num_electrode_arcs = 5         # number of concentric arcs per ring

# === Create main horizontal bus waveguide ===
bus_length = 100  # µm
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2, 0))

# === Create 3 ring resonators coupled to bus ===
rings = []
ring_centers = []

for i in range(num_rings):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    # Position: rings below the bus waveguide
    x_pos = -25 + i * ring_spacing
    y_pos = -(coupling_gap + ring_radius)  # Below the bus
    ring.move((x_pos, y_pos))
    rings.append(ring)
    ring_centers.append((x_pos, y_pos))

# === Create electrode arcs around each ring ===
# The image shows partial arcs (like semi-circles or 270-degree arcs)
# positioned around the top/sides of each ring

def create_arc(radius, width, start_angle, end_angle, layer=(2, 0)):
    """Create an arc (partial ring) component."""
    arc = gf.Component()
    # Convert angles to radians
    start_rad = np.radians(start_angle)
    end_rad = np.radians(end_angle)

    # Create arc using path
    angles = np.linspace(start_rad, end_rad, 100)
    points_outer = [(radius + width/2) * np.cos(a) for a in angles], [(radius + width/2) * np.sin(a) for a in angles]
    points_inner = [(radius - width/2) * np.cos(a) for a in angles], [(radius - width/2) * np.sin(a) for a in angles]

    # Create polygon from inner and outer points
    outer_pts = list(zip(points_outer[0], points_outer[1]))
    inner_pts = list(zip(points_inner[0], points_inner[1]))[::-1]  # Reverse for proper polygon
    all_pts = outer_pts + inner_pts

    arc.add_polygon(all_pts, layer=layer)
    return arc

# Add electrode arcs around each ring
electrode_arc_width = 0.8  # µm
for i, (cx, cy) in enumerate(ring_centers):
    # Create multiple concentric electrode arcs
    for j in range(num_electrode_arcs):
        arc_radius = electrode_inner_radius + j * 0.8
        # Arcs covering the outer portion (avoiding coupling region)
        arc_comp = create_arc(
            radius=arc_radius,
            width=electrode_arc_width,
            start_angle=-135,  # Start from lower-left
            end_angle=135,     # End at lower-right (270 degree arc, open at bottom)
            layer=(2, 0)
        )
        arc_ref = c << arc_comp
        arc_ref.move((cx, cy))

# === Add grating couplers on left edge ===
# The image shows horizontal grating patterns on the left

# Input grating coupler
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Additional grating couplers stacked vertically (I see multiple horizontal lines)
gc_y_positions = [-30, -45, -60]  # Multiple GCs below main bus
for y_pos in gc_y_positions:
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((-bus_length/2 - 20, y_pos))
    gc.rotate(0)  # Point to the right

# === Add metal contact pads at top right ===
# Labeled (iv) and (v) in the image - appears to be 5-6 pads

pad_size = (15, 12)  # µm
pad_x_start = 40
pad_y_start = 30
pad_spacing_y = 15

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x_start, pad_y_start - i * pad_spacing_y))
    pads.append(pad)

# === Add routing traces from pads to electrode arcs ===
# Metal routing connecting pads to the electrode structures

trace_width = 2.0

# Route from each pad toward its corresponding electrode region
for i, pad in enumerate(pads):
    # Horizontal trace from pad
    pad_center = (pad_x_start + pad_size[0]/2, pad_y_start - i * pad_spacing_y + pad_size[1]/2)

    # Route toward the ring area
    trace_h = c << gf.components.rectangle(size=(20, trace_width), layer=(2, 0))
    trace_h.move((pad_center[0] - 25, pad_center[1] - trace_width/2))

    # Vertical trace down toward rings
    trace_v = c << gf.components.rectangle(size=(trace_width, 15 + i * 5), layer=(2, 0))
    trace_v.move((pad_center[0] - 25, pad_center[1] - 15 - i * 5))

# === Add output waveguide with splitter ===
# The right side appears to have a split/branching structure

# Add MMI splitter at right end of bus
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
splitter.connect('o1', bus.ports['o2'])

# Output waveguides from splitter
bend_r = 10.0

# Upper branch
straight_up = c << gf.components.straight(length=10, width=bus_width)
straight_up.connect('o1', splitter.ports['o2'])

bend_up = c << gf.components.bend_circular(radius=bend_r, angle=45)
bend_up.connect('o1', straight_up.ports['o2'])

# Lower branch
straight_down = c << gf.components.straight(length=10, width=bus_width)
straight_down.connect('o1', splitter.ports['o3'])

bend_down = c << gf.components.bend_circular(radius=bend_r, angle=-45)
bend_down.connect('o1', straight_down.ports['o2'])
