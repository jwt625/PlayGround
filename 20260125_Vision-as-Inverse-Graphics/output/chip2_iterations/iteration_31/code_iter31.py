import gdsfactory as gf
import numpy as np

# Claude's Iteration 31: Fine-tuning to match target proportions
# Adjustments based on careful comparison:
# 1. Shift rings slightly left relative to pads
# 2. Adjust aspect ratio to be wider
# 3. Fine-tune routing pattern
# 4. Match number of components more precisely

c = gf.Component("chip2_iter31_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
bus_width = 0.5
ring_spacing = 24.0  # Wider spacing to match target

# Layout - rings more to the left
ring_y = -8
center_x = -5  # Shift rings left

# === Bus waveguide ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 85
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2 + center_x + 10, bus_y))

# === 3 Ring resonators ===
rings = []
ring_centers = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = center_x - ring_spacing + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === Dense concentric electrode rings ===
electrode_start_r = 10.5
electrode_end_r = 15.0
num_electrodes = 9
electrode_width = 0.35

for cx, cy in ring_centers:
    for i in range(num_electrodes):
        r = electrode_start_r + i * (electrode_end_r - electrode_start_r) / (num_electrodes - 1)
        elec = c << gf.components.ring(radius=r, width=electrode_width, layer=(2, 0))
        elec.move((cx, cy))

# === LEFT: Horizontal line gratings (fiber array) ===
grating_x_end = center_x - ring_spacing - ring_radius - 8
grating_y_center = ring_y
grating_line_spacing = 2.0
grating_line_length = 15.0
grating_line_width = 0.8
num_grating_lines = 12

for i in range(num_grating_lines):
    y = grating_y_center + (num_grating_lines/2 - i - 0.5) * grating_line_spacing
    line = c << gf.components.rectangle(
        size=(grating_line_length, grating_line_width),
        layer=(1, 0)
    )
    line.move((grating_x_end - grating_line_length, y - grating_line_width/2))

# Short waveguide from gratings area to bus
wg_to_grating = c << gf.components.straight(length=5, width=bus_width)
wg_to_grating.connect('o2', bus.ports['o1'])

# === Upper waveguide with vertical drops ===
upper_y = bus_y + 28

wg_upper = c << gf.components.straight(length=70, width=bus_width)
wg_upper.move((center_x - 10, upper_y))

# Vertical drops from upper waveguide
for cx, cy in ring_centers:
    drop_len = upper_y - bus_y - 2
    wg_v = c << gf.components.straight(length=drop_len, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - drop_len/2))

# === Metal pads on RIGHT (iv, v area) ===
# Position pads more to the right to match target proportions
pad_x = center_x + ring_spacing + ring_radius + 25
pad_y_start = upper_y + 8
pad_size = (10, 6)
pad_spacing = 7

pads = []
for i in range(7):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads ===
trace_w = 1.0

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace LEFT
    h_len = 15 + i * 5
    trace_h = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    trace_h.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace DOWN
    v_len = py - ring_y + 10
    trace_v = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    trace_v.move((pad_x - h_len - trace_w, py - v_len))

# === Right edge: Vertical hook structures (matching target's v area) ===
right_edge_x = pad_x + pad_size[0] + 8

for i in range(4):
    y_top = upper_y - i * 10
    # Vertical segment
    wg_right = c << gf.components.straight(length=12, width=bus_width)
    wg_right.rotate(90)
    wg_right.move((right_edge_x, y_top - 6))

    # Hook at bottom
    hook = c << gf.components.bend_circular(radius=5, angle=-90)
    hook.move((right_edge_x - 5, y_top - 12))

# === Output from bus ===
out_wg = c << gf.components.straight(length=10, width=bus_width)
out_wg.connect('o1', bus.ports['o2'])

# Upper branch
bend_up = c << gf.components.bend_circular(radius=6, angle=90)
bend_up.connect('o1', out_wg.ports['o2'])

wg_up = c << gf.components.straight(length=12, width=bus_width)
wg_up.connect('o1', bend_up.ports['o2'])

# Lower branch
bend_down = c << gf.components.bend_circular(radius=6, angle=-90)
bend_down.move((bus_length/2 + center_x + 20 - 6, bus_y - 6))

wg_down = c << gf.components.straight(length=12, width=bus_width)
wg_down.connect('o1', bend_down.ports['o2'])
