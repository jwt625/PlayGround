import gdsfactory as gf
import numpy as np

# Claude's Iteration 30: Match target more closely
# Key remaining differences:
# 1. Target right side (v) has vertical hook-like structures
# 2. Need more routing traces from pads going horizontally left
# 3. Electrode patterns should match target's appearance better

c = gf.Component("chip2_iter30_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
bus_width = 0.5
ring_spacing = 22.0

# Layout
ring_y = -8
center_x = 3

# === Bus waveguide ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 75
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
electrode_start_r = 10.5
electrode_end_r = 15.0
num_electrodes = 9
electrode_width = 0.35

for cx, cy in ring_centers:
    for i in range(num_electrodes):
        r = electrode_start_r + i * (electrode_end_r - electrode_start_r) / (num_electrodes - 1)
        elec = c << gf.components.ring(radius=r, width=electrode_width, layer=(2, 0))
        elec.move((cx, cy))

# === LEFT: Horizontal line gratings ===
grating_x_end = -bus_length/2 + center_x - 3
grating_y_center = ring_y
grating_line_spacing = 2.2
grating_line_length = 15.0
grating_line_width = 0.9
num_grating_lines = 12

for i in range(num_grating_lines):
    y = grating_y_center + (num_grating_lines/2 - i - 0.5) * grating_line_spacing
    line = c << gf.components.rectangle(
        size=(grating_line_length, grating_line_width),
        layer=(1, 0)
    )
    line.move((grating_x_end - grating_line_length, y - grating_line_width/2))

# Connect gratings to bus
wg_grating = c << gf.components.straight(length=6, width=bus_width)
wg_grating.connect('o2', bus.ports['o1'])

# === Upper waveguide with vertical drops ===
upper_y = bus_y + 25

wg_upper = c << gf.components.straight(length=60, width=bus_width)
wg_upper.move((center_x - 30, upper_y))

# Vertical drops
for cx, cy in ring_centers:
    drop_len = upper_y - bus_y - 2
    wg_v = c << gf.components.straight(length=drop_len, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - drop_len/2))

# === Metal pads (right side) - matching target's (iv) and (v) ===
pad_x = center_x + ring_spacing + 18
pad_y_start = upper_y + 5
pad_size = (9, 6)
pad_spacing = 7

pads = []
for i in range(7):  # More pads to match target
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads (horizontal + vertical traces) ===
trace_w = 1.0

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace going LEFT from pad
    h_len = 10 + i * 5
    trace_h = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    trace_h.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace going DOWN
    v_len = py - ring_y + 10
    trace_v = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    trace_v.move((pad_x - h_len - trace_w, py - v_len))

# === RIGHT SIDE: Vertical structures (matching target's (v) hooks) ===
# Target shows vertical waveguides/routing on far right

right_edge_x = pad_x + pad_size[0] + 5

# Vertical waveguide segments on right edge
for i in range(4):
    y_top = upper_y - i * 12
    wg_right = c << gf.components.straight(length=15, width=bus_width)
    wg_right.rotate(90)
    wg_right.move((right_edge_x, y_top - 7.5))

    # Small hook at bottom of each
    hook = c << gf.components.bend_circular(radius=5, angle=-90)
    hook.move((right_edge_x - 5, y_top - 15))

# === Output from bus (right side) ===
out_wg = c << gf.components.straight(length=8, width=bus_width)
out_wg.connect('o1', bus.ports['o2'])

# Split up and down
bend_up = c << gf.components.bend_circular(radius=6, angle=90)
bend_up.connect('o1', out_wg.ports['o2'])

wg_up = c << gf.components.straight(length=15, width=bus_width)
wg_up.connect('o1', bend_up.ports['o2'])

# Down branch
wg_split = c << gf.components.straight(length=3, width=bus_width)
wg_split.move((bus_length/2 + center_x + 8, bus_y))

bend_down = c << gf.components.bend_circular(radius=6, angle=-90)
bend_down.move((bus_length/2 + center_x + 11, bus_y - 6))

wg_down = c << gf.components.straight(length=15, width=bus_width)
wg_down.connect('o1', bend_down.ports['o2'])
