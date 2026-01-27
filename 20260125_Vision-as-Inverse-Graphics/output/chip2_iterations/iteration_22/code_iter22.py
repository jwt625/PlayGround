import gdsfactory as gf
import numpy as np

# Claude's Iteration 22: Refined based on comparison
# Key refinements from viewing target vs iter21:
# 1. Rings should be positioned in lower half, coupled to bus above them
# 2. Electrode rings should be more concentric (full circles like iter9)
# 3. Grating couplers need to be horizontal gratings on left edge
# 4. More complex waveguide topology at top
# 5. Metal routing from pads down to electrode rings

c = gf.Component("chip2_iter22_claude")

# === Parameters ===
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20       # From iter9 analysis
ring_spacing = 20.0       # Spread rings more
bus_width = 0.5
num_rings = 3

# === Layout origin - position rings in lower half ===
layout_center_x = 0
ring_y = -20  # Rings in lower half

# === Create 3 ring resonators ===
rings = []
ring_centers = []

for i in range(num_rings):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    x_pos = layout_center_x - ring_spacing + i * ring_spacing
    ring.move((x_pos, ring_y))
    rings.append(ring)
    ring_centers.append((x_pos, ring_y))

# === Bus waveguide above rings for coupling ===
# Each ring coupled to its own bus segment
bus_y = ring_y + ring_radius + coupling_gap  # Just above rings

# Create continuous bus waveguide
bus_length = 80
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.move((-bus_length/2, bus_y))

# === Concentric electrode rings (full circles like iter9) ===
# Matching iter9's successful configuration
electrode_radii = [11.0, 10.75, 10.5, 10.25, 10.0, 9.75, 9.5, 9.25, 9.0]
electrode_width = 0.3

for cx, cy in ring_centers:
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move((cx, cy))

# === Grating couplers on left edge ===
# Positioned as a vertical array, pointing right into waveguides

gc_x = -bus_length/2 - 10  # Left of bus

# Main input GC connected to bus
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Additional GCs below the main one (matching the horizontal line pattern in target)
gc_spacing = 12.0
for i in range(3):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc_y = ring_y - 15 - i * gc_spacing
    gc.move((gc_x, gc_y))

# === Upper waveguide routing structure ===
# The target shows waveguides at the top connecting to structures

upper_wg_y = bus_y + 20

# Horizontal waveguide at top
wg_upper = c << gf.components.straight(length=60, width=bus_width)
wg_upper.move((-30, upper_wg_y))

# Vertical connections from upper waveguide down to rings region
for i, (cx, cy) in enumerate(ring_centers):
    # Vertical waveguide segment
    wg_v = c << gf.components.straight(length=15, width=bus_width)
    wg_v.rotate(90)  # Make vertical
    wg_v.move((cx, upper_wg_y - 7.5))

    # Bend connecting to horizontal
    bend = c << gf.components.bend_circular(radius=5, angle=90)
    bend.move((cx - 5, upper_wg_y))

# === Metal contact pads at top right ===
# Arranged vertically as seen in target (iv, v labels)

pad_size = (12, 10)
pad_x = 45
pad_y_start = upper_wg_y + 15
pad_spacing = 12

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing from pads to electrode rings ===
# Traces running from pads down and left to connect to electrode structures

trace_width = 1.5

# Route traces from each pad toward the ring electrode region
for i, pad in enumerate(pads):
    pad_y = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace from pad going left
    h_length = 20 + i * 8  # Staggered lengths
    trace_h = c << gf.components.rectangle(size=(h_length, trace_width), layer=(2, 0))
    trace_h.move((pad_x - h_length, pad_y - trace_width/2))

    # Vertical trace going down toward rings
    v_length = pad_y - ring_y + 5
    trace_v = c << gf.components.rectangle(size=(trace_width, v_length), layer=(2, 0))
    trace_v.move((pad_x - h_length, pad_y - v_length))

# === Output structure on right ===
# MMI splitter at end of bus

splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
splitter.connect('o1', bus.ports['o2'])

# Output bends
bend_r = 10.0

straight1 = c << gf.components.straight(length=8, width=bus_width)
straight1.connect('o1', splitter.ports['o2'])

bend1 = c << gf.components.bend_circular(radius=bend_r, angle=90)
bend1.connect('o1', straight1.ports['o2'])

straight2 = c << gf.components.straight(length=8, width=bus_width)
straight2.connect('o1', splitter.ports['o3'])

bend2 = c << gf.components.bend_circular(radius=bend_r, angle=-90)
bend2.connect('o1', straight2.ports['o2'])

# Final output GCs
gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', bend1.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', bend2.ports['o2'])
