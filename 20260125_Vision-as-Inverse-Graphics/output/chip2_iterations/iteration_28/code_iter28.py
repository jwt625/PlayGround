import gdsfactory as gf
import numpy as np

# Claude's Iteration 28: Addressing actual visual differences
# Key observations from target chip2.png:
# 1. LEFT: Horizontal LINE gratings (not triangular GCs) - multiple parallel lines
# 2. RINGS: 3 rings with TIGHT spiral/concentric electrode patterns
# 3. CENTER: Vertical waveguide segments connecting regions
# 4. RIGHT: Metal pads (iv, v) with routing traces going left toward rings
# 5. The electrode pattern looks like dense concentric circles or spirals

c = gf.Component("chip2_iter28_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
bus_width = 0.5

# Ring spacing - looking at target, rings are evenly spaced
ring_spacing = 25.0  # Appears wider in target

# Layout positioning
ring_y = -10
center_x = 0

# === Create horizontal bus waveguide (above rings) ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 85
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2, bus_y))

# === 3 Ring resonators ===
rings = []
ring_centers = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = center_x - ring_spacing + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === TIGHT concentric electrode rings (matching target's dense pattern) ===
# Target shows very dense concentric patterns - many thin rings close together
electrode_start_r = 10.5
electrode_end_r = 16.0
num_electrodes = 12  # More rings, tighter spacing
electrode_width = 0.25  # Thinner

for cx, cy in ring_centers:
    for i in range(num_electrodes):
        r = electrode_start_r + i * (electrode_end_r - electrode_start_r) / num_electrodes
        elec = c << gf.components.ring(radius=r, width=electrode_width, layer=(2, 0))
        elec.move((cx, cy))

# === LEFT SIDE: Horizontal LINE gratings (NOT elliptical GCs) ===
# Target shows horizontal parallel lines - create simple line grating pattern
grating_x = -bus_length/2 - 5
grating_y_start = ring_y + 10
grating_line_spacing = 2.0
grating_line_length = 15.0
grating_line_width = 0.8
num_grating_lines = 12

for i in range(num_grating_lines):
    y = grating_y_start - i * grating_line_spacing
    line = c << gf.components.rectangle(
        size=(grating_line_length, grating_line_width),
        layer=(1, 0)
    )
    line.move((grating_x - grating_line_length, y))

# Connect gratings to bus with waveguide
wg_to_grating = c << gf.components.straight(length=10, width=bus_width)
wg_to_grating.connect('o2', bus.ports['o1'])

# === VERTICAL waveguide segments (center region) ===
# Target shows clear vertical waveguides connecting top and bottom
upper_y = bus_y + 30
lower_y = ring_y - ring_radius - 10

# Upper horizontal waveguide
wg_upper = c << gf.components.straight(length=70, width=bus_width)
wg_upper.move((-35, upper_y))

# Vertical drops from upper waveguide to ring region
for cx, cy in ring_centers:
    # Vertical segment
    vert_length = upper_y - bus_y - 5
    wg_v = c << gf.components.straight(length=vert_length, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - vert_length/2))

# Lower horizontal waveguide
wg_lower = c << gf.components.straight(length=70, width=bus_width)
wg_lower.move((-35, lower_y))

# Vertical segments from rings to lower waveguide
for cx, cy in ring_centers:
    vert_length = cy - ring_radius - coupling_gap - lower_y
    wg_v_low = c << gf.components.straight(length=vert_length, width=bus_width)
    wg_v_low.rotate(90)
    wg_v_low.move((cx, lower_y + vert_length/2))

# === RIGHT SIDE: Metal pads (iv, v) with routing ===
pad_x = 50
pad_y_start = upper_y + 5
pad_size = (12, 8)
pad_spacing = 10

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads toward electrode rings ===
# Target shows routing going LEFT and DOWN from pads
trace_w = 1.5

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace going LEFT from pad
    h_len = 15 + i * 8
    trace_h = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    trace_h.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace going DOWN toward electrodes
    v_len = py - ring_y + 5
    trace_v = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    trace_v.move((pad_x - h_len - trace_w, py - v_len))

# === RIGHT side output ===
# Simple output structure
out_x = bus_length/2 + 5
out_wg = c << gf.components.straight(length=15, width=bus_width)
out_wg.connect('o1', bus.ports['o2'])

# Output splits going up and down (matching target's structure on right)
bend_up = c << gf.components.bend_circular(radius=10, angle=90)
bend_up.connect('o1', out_wg.ports['o2'])

straight_up = c << gf.components.straight(length=20, width=bus_width)
straight_up.connect('o1', bend_up.ports['o2'])

bend_down = c << gf.components.bend_circular(radius=10, angle=-90)
bend_down.move((out_x + 15, bus_y))

straight_down = c << gf.components.straight(length=20, width=bus_width)
straight_down.connect('o1', bend_down.ports['o2'])
