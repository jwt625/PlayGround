import gdsfactory as gf
import numpy as np

# Claude's Iteration 24: Combining best approaches
# - Full concentric electrode rings (like successful iter9)
# - Thicker electrodes for visibility (like iter23)
# - Clean layout matching target topology
# - Proper grating coupler placement

c = gf.Component("chip2_iter24_claude")

# === Parameters (using iter9's successful values with adjustments) ===
ring_radius = 10.0        # µm
ring_width = 0.5          # µm
coupling_gap = 0.20       # µm (iter9 optimal)
ring_spacing = 20.0       # µm between ring centers
bus_width = 0.5           # µm

# === Coordinate system ===
# Rings centered horizontally, positioned in lower portion
center_x = 0
ring_y = -10

# === Create horizontal bus waveguide ===
bus_length = 80
bus_y = ring_y + ring_radius + coupling_gap
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2, bus_y))

# === Create 3 ring resonators ===
rings = []
ring_centers = []

for i in range(3):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    x_pos = center_x - ring_spacing + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === Concentric electrode rings around each resonator ===
# Using thicker, more visible rings
electrode_radii = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]  # Wider spacing
electrode_width = 0.6  # Thicker for visibility

for cx, cy in ring_centers:
    for elec_r in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_r,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move((cx, cy))

# === Input grating coupler (left) ===
gc_in = c << gf.components.grating_coupler_elliptical_te()
gc_in.connect('o1', bus.ports['o1'])

# === Additional grating couplers in vertical array (matching target left edge) ===
# Target shows horizontal grating lines on left
gc_left_x = -bus_length/2 - 15 - 10  # Position left of input GC

for i in range(3):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_left_x - 5, ring_y - 20 - i * 12))

# === Upper waveguide structure ===
# Target shows waveguides at top connecting downward
upper_y = bus_y + 30

wg_upper = c << gf.components.straight(length=60, width=bus_width)
wg_upper.move((-30, upper_y))

# Vertical connections from upper waveguide
for i, (cx, cy) in enumerate(ring_centers):
    # Vertical drop
    wg_v = c << gf.components.straight(length=25, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, upper_y - 12.5))

# === Metal contact pads (right side, vertically arranged) ===
pad_size = (12, 10)
pad_x = 50
pad_y_start = upper_y + 5
pad_spacing = 13

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing traces ===
trace_w = 1.5

# Connect pads to electrode regions with L-shaped traces
for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal segment from pad
    h_len = 20 + i * 8
    th = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    th.move((pad_x - h_len, py - trace_w/2))

    # Vertical segment down
    v_len = py - ring_y + 15
    tv = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    tv.move((pad_x - h_len - trace_w, py - v_len))

# === Output MMI splitter and routing ===
mmi = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
mmi.connect('o1', bus.ports['o2'])

# Upper output branch
s1 = c << gf.components.straight(length=8, width=bus_width)
s1.connect('o1', mmi.ports['o2'])

b1 = c << gf.components.bend_circular(radius=10, angle=90)
b1.connect('o1', s1.ports['o2'])

s2 = c << gf.components.straight(length=12, width=bus_width)
s2.connect('o1', b1.ports['o2'])

gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', s2.ports['o2'])

# Lower output branch
s3 = c << gf.components.straight(length=8, width=bus_width)
s3.connect('o1', mmi.ports['o3'])

b2 = c << gf.components.bend_circular(radius=10, angle=-90)
b2.connect('o1', s3.ports['o2'])

s4 = c << gf.components.straight(length=12, width=bus_width)
s4.connect('o1', b2.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', s4.ports['o2'])

# === Additional output grating couplers (matching target) ===
# Target shows multiple GCs on right side too
for i in range(2):
    gc_extra = c << gf.components.grating_coupler_elliptical_te()
    gc_extra.move((pad_x + 20, ring_y - 25 - i * 15))
