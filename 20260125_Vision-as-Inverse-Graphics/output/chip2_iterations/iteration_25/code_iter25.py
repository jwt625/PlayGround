import gdsfactory as gf
import numpy as np

# Claude's Iteration 25: Addressing VLM feedback
# Key fixes:
# 1. Left grating couplers in HORIZONTAL array (not vertical)
# 2. Smaller electrode dimensions (width 0.35µm, radii 10.5-14.5µm)
# 3. Proper vertical drops connecting upper waveguide to ring coupling region
# 4. Both MMI outputs properly connected

c = gf.Component("chip2_iter25_claude")

# === Parameters (refined per VLM feedback) ===
ring_radius = 10.0        # µm (confirmed correct)
ring_width = 0.5          # µm (confirmed correct)
coupling_gap = 0.25       # µm (adjusted per feedback)
ring_spacing = 20.0       # µm (confirmed correct)
bus_width = 0.5           # µm (keep at 0.5 to match grating coupler)

# === Layout positioning ===
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

# === Concentric electrode rings (adjusted dimensions per VLM) ===
electrode_radii = [10.5, 11.5, 12.5, 13.5, 14.5]  # Reduced per feedback
electrode_width = 0.35  # Reduced per feedback

for cx, cy in ring_centers:
    for elec_r in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_r,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move((cx, cy))

# === Input grating coupler connected to bus ===
gc_in = c << gf.components.grating_coupler_elliptical_te()
gc_in.connect('o1', bus.ports['o1'])

# === Additional grating couplers in HORIZONTAL array (key fix!) ===
# Target shows horizontal line patterns - GCs spaced horizontally at same y level
gc_array_y = ring_y - 25  # Below the rings
gc_array_x_start = -bus_length/2 - 10

for i in range(4):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_array_x_start - i * 15, gc_array_y))

# === Upper waveguide with PROPER vertical drops to ring coupling ===
upper_y = bus_y + 35

wg_upper = c << gf.components.straight(length=70, width=bus_width)
wg_upper.move((-35, upper_y))

# Vertical drops connecting upper waveguide DOWN to bus waveguide region
for i, (cx, cy) in enumerate(ring_centers):
    # Vertical straight from upper waveguide toward bus
    drop_length = upper_y - bus_y - 5
    wg_drop = c << gf.components.straight(length=drop_length, width=bus_width)
    wg_drop.rotate(90)
    wg_drop.move((cx, upper_y - drop_length/2))

    # Add bends at top connecting to horizontal
    bend_top = c << gf.components.bend_circular(radius=5, angle=90)
    bend_top.move((cx - 5, upper_y))

# === Metal contact pads (right side) ===
pad_size = (12, 10)
pad_x = 55
pad_y_start = upper_y
pad_spacing = 12

pads = []
for i in range(5):
    pad = c << gf.components.rectangle(size=pad_size, layer=(2, 0))
    pad.move((pad_x, pad_y_start - i * pad_spacing))
    pads.append(pad)

# === Metal routing traces (improved pattern) ===
trace_w = 1.2

for i, pad in enumerate(pads):
    py = pad_y_start - i * pad_spacing + pad_size[1]/2

    # Horizontal trace from pad going left
    h_len = 15 + i * 6
    th = c << gf.components.rectangle(size=(h_len, trace_w), layer=(2, 0))
    th.move((pad_x - h_len, py - trace_w/2))

    # Vertical trace going down toward electrodes
    v_len = py - ring_y + 10
    tv = c << gf.components.rectangle(size=(trace_w, v_len), layer=(2, 0))
    tv.move((pad_x - h_len - trace_w, py - v_len))

# === MMI splitter with BOTH outputs properly connected ===
mmi = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=5, length_mmi=8)
mmi.connect('o1', bus.ports['o2'])

# Output 1 (upper branch)
s1 = c << gf.components.straight(length=10, width=bus_width)
s1.connect('o1', mmi.ports['o2'])

b1 = c << gf.components.bend_circular(radius=10, angle=90)
b1.connect('o1', s1.ports['o2'])

s1b = c << gf.components.straight(length=15, width=bus_width)
s1b.connect('o1', b1.ports['o2'])

gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', s1b.ports['o2'])

# Output 2 (lower branch) - explicitly connected
s2 = c << gf.components.straight(length=10, width=bus_width)
s2.connect('o1', mmi.ports['o3'])

b2 = c << gf.components.bend_circular(radius=10, angle=-90)
b2.connect('o1', s2.ports['o2'])

s2b = c << gf.components.straight(length=15, width=bus_width)
s2b.connect('o1', b2.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', s2b.ports['o2'])

# === Additional output grating couplers on right (per feedback) ===
for i in range(3):
    gc_extra = c << gf.components.grating_coupler_elliptical_te()
    gc_extra.move((pad_x + 15, ring_y - 20 - i * 15))
