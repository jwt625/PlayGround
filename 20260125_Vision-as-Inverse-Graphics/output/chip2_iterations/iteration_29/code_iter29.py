import gdsfactory as gf
import numpy as np

# Claude's Iteration 29: Further refinement
# Comparing iter28 to target:
# - Remove lower horizontal waveguide (not in target)
# - Adjust ring positions to better match target layout
# - Fix right side output structure
# - Improve proportions

c = gf.Component("chip2_iter29_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
bus_width = 0.5
ring_spacing = 22.0  # Adjusted to better match target spacing

# Layout - rings in lower-center area
ring_y = -8
center_x = 5  # Slight shift right to match target

# === Horizontal bus waveguide (main bus above rings) ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 80
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

# === Dense concentric electrode rings ===
# Target shows tight concentric patterns
electrode_start_r = 10.5
electrode_end_r = 15.5
num_electrodes = 10
electrode_width = 0.3

for cx, cy in ring_centers:
    for i in range(num_electrodes):
        r = electrode_start_r + i * (electrode_end_r - electrode_start_r) / (num_electrodes - 1)
        elec = c << gf.components.ring(radius=r, width=electrode_width, layer=(2, 0))
        elec.move((cx, cy))

# === LEFT: Horizontal line gratings ===
# Multiple parallel horizontal lines (fiber array interface)
grating_x_end = -bus_length/2 + center_x - 5
grating_y_center = ring_y
grating_line_spacing = 2.5
grating_line_length = 18.0
grating_line_width = 1.0
num_grating_lines = 10

for i in range(num_grating_lines):
    y = grating_y_center + (num_grating_lines/2 - i) * grating_line_spacing
    line = c << gf.components.rectangle(
        size=(grating_line_length, grating_line_width),
        layer=(1, 0)
    )
    line.move((grating_x_end - grating_line_length, y - grating_line_width/2))

# Waveguide connecting gratings to bus
wg_grating = c << gf.components.straight(length=8, width=bus_width)
wg_grating.connect('o2', bus.ports['o1'])

# === Upper waveguide with vertical drops ===
upper_y = bus_y + 28

wg_upper = c << gf.components.straight(length=65, width=bus_width)
wg_upper.move((center_x - 32.5, upper_y))

# Vertical drops from upper waveguide
for cx, cy in ring_centers:
    drop_len = upper_y - bus_y - 3
    wg_v = c << gf.components.straight(length=drop_len, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - drop_len/2))

# === Metal pads on RIGHT (iv, v in target) ===
pad_x = center_x + ring_spacing + 20
pad_y_start = upper_y + 3
pad_size = (10, 7)
pad_spacing = 8

pads = []
for i in range(6):  # 6 pads to better match target
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads to electrodes ===
trace_w = 1.2

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace LEFT
    h_len = 12 + i * 6
    trace_h = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    trace_h.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace DOWN
    v_len = py - ring_y + 8
    trace_v = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    trace_v.move((pad_x - h_len - trace_w, py - v_len))

# === RIGHT side: Output waveguides ===
# Target shows waveguides going up and down on right side

# Main output from bus
out_straight = c << gf.components.straight(length=10, width=bus_width)
out_straight.connect('o1', bus.ports['o2'])

# Upper branch
bend_up = c << gf.components.bend_circular(radius=8, angle=90)
bend_up.connect('o1', out_straight.ports['o2'])

wg_up = c << gf.components.straight(length=18, width=bus_width)
wg_up.connect('o1', bend_up.ports['o2'])

# Lower branch (from same point)
straight_before_down = c << gf.components.straight(length=3, width=bus_width)
straight_before_down.move((bus_length/2 + center_x + 10, bus_y))

bend_down = c << gf.components.bend_circular(radius=8, angle=-90)
bend_down.move((bus_length/2 + center_x + 13, bus_y - 8))

wg_down = c << gf.components.straight(length=18, width=bus_width)
wg_down.connect('o1', bend_down.ports['o2'])
