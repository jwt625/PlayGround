import gdsfactory as gf
import numpy as np

# Claude's Iteration 23: Focus on visible electrode structure
# Key changes:
# 1. Thicker electrode rings for better visibility
# 2. Electrode arcs open at coupling region (bottom)
# 3. Simpler, cleaner layout matching target topology
# 4. Rings positioned in lower-center region

c = gf.Component("chip2_iter23_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
ring_spacing = 22.0       # Spacing between ring centers
bus_width = 0.5
num_rings = 3

# === Ring positions (center of layout) ===
ring_y = -15  # Rings in lower portion
ring_x_start = -ring_spacing  # Center the 3 rings

# === Create bus waveguide ABOVE the rings ===
bus_y = ring_y + ring_radius + coupling_gap + bus_width/2
bus_length = 90
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2, bus_y))

# === Create 3 ring resonators ===
rings = []
ring_centers = []

for i in range(num_rings):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    x_pos = ring_x_start + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === Create electrode arcs (partial rings) around each resonator ===
# These should be visible and NOT cover the coupling region at the top

def create_arc_polygon(radius, width, start_angle, end_angle, layer=(2, 0)):
    """Create an arc as a polygon."""
    comp = gf.Component()
    n_points = 50

    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), n_points)

    # Outer points
    outer_r = radius + width/2
    outer_pts = [(outer_r * np.cos(a), outer_r * np.sin(a)) for a in angles]

    # Inner points (reversed)
    inner_r = radius - width/2
    inner_pts = [(inner_r * np.cos(a), inner_r * np.sin(a)) for a in angles[::-1]]

    all_pts = outer_pts + inner_pts
    comp.add_polygon(all_pts, layer=layer)
    return comp

# Electrode parameters - make them thick and visible
electrode_base_radius = ring_radius + 2.0  # Start outside the ring
electrode_width = 0.8  # Thicker for visibility
num_electrode_rings = 7  # Number of concentric electrode arcs
electrode_spacing = 1.0  # Spacing between electrode rings

for cx, cy in ring_centers:
    for j in range(num_electrode_rings):
        arc_radius = electrode_base_radius + j * electrode_spacing
        # Arc covering bottom 270 degrees (leaving top open for light coupling)
        # Start at -225 degrees, end at +45 degrees (270 degree arc)
        arc = create_arc_polygon(
            radius=arc_radius,
            width=electrode_width,
            start_angle=-225,  # Lower left
            end_angle=45,      # Upper right
            layer=(2, 0)
        )
        arc_ref = c << arc
        arc_ref.move((cx, cy))

# === Grating couplers on LEFT edge ===
# Input GC connected to bus
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Additional GCs in vertical array (matching horizontal line pattern in target)
gc_left_x = -bus_length/2 - 25
gc_y_positions = [ring_y, ring_y - 15, ring_y - 30]
for y_pos in gc_y_positions:
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_left_x, y_pos))

# === Waveguide routing at top ===
upper_y = bus_y + 25

# Horizontal waveguide at top
wg_top = c << gf.components.straight(length=70, width=bus_width)
wg_top.move((-35, upper_y))

# Vertical drops connecting top waveguide to ring region
for i, (cx, cy) in enumerate(ring_centers):
    wg_drop = c << gf.components.straight(length=20, width=bus_width)
    wg_drop.rotate(90)
    wg_drop.move((cx, upper_y - 10))

# === Metal contact pads at top right ===
pad_size = (15, 12)
pad_x = 55
pad_y_start = upper_y + 10
pad_spacing = 14

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads to electrode structures ===
trace_width = 2.0

for i, pad in enumerate(pads):
    pad_y = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace from pad to left
    h_len = 15 + i * 10
    trace_h = c << gf.components.rectangle(size=(h_len, trace_width), layer=(2, 0))
    trace_h.move((pad_x - h_len, pad_y - trace_width/2))

    # Vertical trace down toward ring electrodes
    v_len = pad_y - ring_y + 10
    trace_v = c << gf.components.rectangle(size=(trace_width, v_len), layer=(2, 0))
    trace_v.move((pad_x - h_len - trace_width, ring_y - 5))

# === Output structure on right ===
# Simple output with bends to GCs

splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
splitter.connect('o1', bus.ports['o2'])

# Upper output path
s_up = c << gf.components.straight(length=10, width=bus_width)
s_up.connect('o1', splitter.ports['o2'])

b_up = c << gf.components.bend_circular(radius=10, angle=90)
b_up.connect('o1', s_up.ports['o2'])

s_up2 = c << gf.components.straight(length=15, width=bus_width)
s_up2.connect('o1', b_up.ports['o2'])

gc_out_up = c << gf.components.grating_coupler_elliptical_te()
gc_out_up.connect('o1', s_up2.ports['o2'])

# Lower output path
s_down = c << gf.components.straight(length=10, width=bus_width)
s_down.connect('o1', splitter.ports['o3'])

b_down = c << gf.components.bend_circular(radius=10, angle=-90)
b_down.connect('o1', s_down.ports['o2'])

s_down2 = c << gf.components.straight(length=15, width=bus_width)
s_down2.connect('o1', b_down.ports['o2'])

gc_out_down = c << gf.components.grating_coupler_elliptical_te()
gc_out_down.connect('o1', s_down2.ports['o2'])
