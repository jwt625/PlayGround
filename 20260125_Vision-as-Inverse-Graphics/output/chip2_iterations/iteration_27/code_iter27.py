import gdsfactory as gf
import numpy as np

# Claude's Iteration 27: Closest match to target topology
# Based on careful analysis of chip2.png:
# - 3 ring resonators (labeled ii) with partial electrode arcs
# - Horizontal grating couplers on left (fiber array)
# - Waveguide routing at top (labeled i)
# - Metal pads (iv, v) at top right with routing
# - Proportions more closely matching the wide aspect ratio

c = gf.Component("chip2_iter27_claude")

# === Parameters matching target observations ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
ring_spacing = 20.0
bus_width = 0.5

# Layout - wider aspect ratio to match target
ring_y = -5
center_x = 5  # Shift rings slightly right of center

# === Horizontal bus waveguide ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 65
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2 + center_x, bus_y))

# === 3 Ring resonators ===
rings = []
ring_centers = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = center_x - ring_spacing + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === Electrode arcs (partial circles around rings) ===
# Target shows concentric structures - use partial arcs
def create_arc(radius, width, start_angle, end_angle, n_pts=50, layer=(2, 0)):
    """Create arc polygon."""
    comp = gf.Component()
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), n_pts)
    outer_r = radius + width/2
    inner_r = radius - width/2
    outer_pts = [(outer_r * np.cos(a), outer_r * np.sin(a)) for a in angles]
    inner_pts = [(inner_r * np.cos(a), inner_r * np.sin(a)) for a in angles[::-1]]
    comp.add_polygon(outer_pts + inner_pts, layer=layer)
    return comp

# Multiple concentric electrode arcs around each ring
electrode_radii = [11.0, 12.0, 13.0, 14.0, 15.0]
electrode_width = 0.5

for cx, cy in ring_centers:
    for r in electrode_radii:
        # Arc from -135° to +135° (270° arc, open at bottom for coupling)
        arc = create_arc(r, electrode_width, -135, 135, layer=(2, 0))
        arc_ref = c << arc
        arc_ref.move((cx, cy))

# === Grating couplers on LEFT as horizontal array ===
# Target shows horizontal gratings - place them in a column pointing right
gc_array_x = -bus_length/2 + center_x - 15  # Left edge
gc_array_y_start = ring_y + 5
gc_spacing_y = 8

# Input GC connected to bus
gc_in = c << gf.components.grating_coupler_elliptical_te()
gc_in.connect('o1', bus.ports['o1'])

# Additional GCs in vertical column (horizontal orientation)
for i in range(4):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_array_x, gc_array_y_start - i * gc_spacing_y))

# === Upper waveguide routing (labeled i in target) ===
upper_y = bus_y + 25

# Main horizontal waveguide
wg_upper = c << gf.components.straight(length=50, width=bus_width)
wg_upper.move((center_x - 25, upper_y))

# Vertical drops to ring region
for i, (cx, cy) in enumerate(ring_centers):
    drop_len = upper_y - bus_y - 3
    wg_v = c << gf.components.straight(length=drop_len, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - drop_len/2))

# === Metal contact pads (iv, v in target) - right side ===
pad_x = center_x + ring_spacing + 25
pad_y_start = upper_y
pad_size = (10, 8)
pad_spacing = 9

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing traces (L-shaped from pads to electrodes) ===
trace_w = 1.0

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace left
    h_len = 10 + i * 5
    th = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    th.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace down
    v_len = py - ring_y + 8
    tv = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    tv.move((pad_x - h_len - trace_w, ring_y - 5))

# === Output structure on right ===
# MMI splitter with two outputs
mmi = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
mmi.connect('o1', bus.ports['o2'])

# Upper output
s1 = c << gf.components.straight(length=8, width=bus_width)
s1.connect('o1', mmi.ports['o2'])
b1 = c << gf.components.bend_circular(radius=10, angle=90)
b1.connect('o1', s1.ports['o2'])
s1b = c << gf.components.straight(length=8, width=bus_width)
s1b.connect('o1', b1.ports['o2'])
gc1 = c << gf.components.grating_coupler_elliptical_te()
gc1.connect('o1', s1b.ports['o2'])

# Lower output
s2 = c << gf.components.straight(length=8, width=bus_width)
s2.connect('o1', mmi.ports['o3'])
b2 = c << gf.components.bend_circular(radius=10, angle=-90)
b2.connect('o1', s2.ports['o2'])
s2b = c << gf.components.straight(length=8, width=bus_width)
s2b.connect('o1', b2.ports['o2'])
gc2 = c << gf.components.grating_coupler_elliptical_te()
gc2.connect('o1', s2b.ports['o2'])
