import gdsfactory as gf
import numpy as np

# Claude's Iteration 26: Simplified focus on core structure
# Based on careful analysis of chip2.png:
# - 3 ring resonators in lower-center region
# - Horizontal grating lines on LEFT edge (fiber array)
# - Concentric electrode structures around rings
# - Metal pads at top-right with routing down
# - Waveguides connecting everything

c = gf.Component("chip2_iter26_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20
ring_spacing = 18.0  # Closer spacing to match target
bus_width = 0.5

# === Layout ===
ring_y = -8
center_x = 0

# === Bus waveguide (above rings for coupling) ===
bus_y = ring_y + ring_radius + coupling_gap
bus_length = 70
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

# === Electrode rings (concentric, tight spacing) ===
# Target shows multiple concentric rings around each resonator
electrode_radii = [10.8, 11.4, 12.0, 12.6, 13.2, 13.8, 14.4]  # 7 rings
electrode_width = 0.4

for cx, cy in ring_centers:
    for r in electrode_radii:
        elec = c << gf.components.ring(radius=r, width=electrode_width, layer=(2, 0))
        elec.move((cx, cy))

# === Left-side grating couplers (HORIZONTAL ARRAY - pointing right) ===
# These form the fiber array interface
gc_x = -bus_length/2 - 12  # Left edge position
gc_y_center = bus_y  # Centered at bus level
gc_y_spacing = 10.0

# Create vertical array of GCs all pointing right
# First connect to bus
gc_bus = c << gf.components.grating_coupler_elliptical_te()
gc_bus.connect('o1', bus.ports['o1'])

# Additional GCs below in fiber array pattern
for i in range(4):
    gc = c << gf.components.grating_coupler_elliptical_te()
    y_offset = -(i + 1) * gc_y_spacing
    gc.move((gc_x, gc_y_center + y_offset))

# === Upper waveguide structure with drops ===
upper_y = bus_y + 30
wg_upper = c << gf.components.straight(length=55, width=bus_width)
wg_upper.move((-27.5, upper_y))

# Vertical drops from upper waveguide
for cx, cy in ring_centers:
    drop_h = upper_y - bus_y - 5
    wg_v = c << gf.components.straight(length=drop_h, width=bus_width)
    wg_v.rotate(90)
    wg_v.move((cx, (upper_y + bus_y) / 2 + 2.5))

# === Metal contact pads (right side, vertical column) ===
pad_x = 45
pad_y_start = upper_y + 5
pad_size = (10, 8)
pad_spacing = 10

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads to electrodes ===
trace_w = 1.0

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace left from pad
    h_len = 12 + i * 5
    th = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    th.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace down toward electrodes
    v_len = py - ring_y + 8
    tv = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    tv.move((pad_x - h_len - trace_w, ring_y - 5))

# === Output MMI splitter with both outputs ===
mmi = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
mmi.connect('o1', bus.ports['o2'])

# Output 1 (up)
s1 = c << gf.components.straight(length=8, width=bus_width)
s1.connect('o1', mmi.ports['o2'])
b1 = c << gf.components.bend_circular(radius=10, angle=90)
b1.connect('o1', s1.ports['o2'])
s1b = c << gf.components.straight(length=10, width=bus_width)
s1b.connect('o1', b1.ports['o2'])
gc1 = c << gf.components.grating_coupler_elliptical_te()
gc1.connect('o1', s1b.ports['o2'])

# Output 2 (down)
s2 = c << gf.components.straight(length=8, width=bus_width)
s2.connect('o1', mmi.ports['o3'])
b2 = c << gf.components.bend_circular(radius=10, angle=-90)
b2.connect('o1', s2.ports['o2'])
s2b = c << gf.components.straight(length=10, width=bus_width)
s2b.connect('o1', b2.ports['o2'])
gc2 = c << gf.components.grating_coupler_elliptical_te()
gc2.connect('o1', s2b.ports['o2'])
