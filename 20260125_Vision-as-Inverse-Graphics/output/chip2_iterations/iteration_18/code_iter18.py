import gdsfactory as gf

# Iteration 18: Based on iteration 9 (75%) with targeted fixes from VLM analysis
# VLM said to keep: 3 rings, input/output routing, tapers
# VLM said to fix: ring geometry (perfectly circular), output symmetry, taper smoothness

c = gf.Component("chip2_iter18")

# Parameters from iteration 9 (which worked best)
ring_radius = 10.0  # µm - VLM confirmed this is correct
ring_width = 0.5    # µm
coupling_gap = 0.20 # µm - iteration 9's value (better than 0.25)
ring_spacing = 13.0 # µm
bus_width = 0.5     # µm

# Create continuous bus waveguide
bus_length = 70  # µm
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)

# Create three PERFECTLY CIRCULAR rings (VLM: fix ring geometry)
rings = []
for i in range(3):
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    x_pos = 12 + i * ring_spacing
    y_pos = bus_width / 2 + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add concentric electrodes (keep from iteration 9)
electrode_radii = [11.0, 10.75, 10.5, 10.25, 10.0, 9.75, 9.5, 9.25, 9.0]
electrode_width = 0.3

for i, ring in enumerate(rings):
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move(ring.center)

# Input GC with SMOOTH taper (VLM: fix taper smoothness)
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Splitter
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=10, length_mmi=10)
splitter.connect('o1', bus.ports['o2'])

# VLM: Fix output symmetry - make both branches identical
bend_radius = 10.0

# Upper output - SYMMETRIC
straight_up1 = c << gf.components.straight(length=5, width=bus_width)
straight_up1.connect('o1', splitter.ports['o2'])

bend_up = c << gf.components.bend_circular(radius=bend_radius, angle=90)
bend_up.connect('o1', straight_up1.ports['o2'])

straight_up2 = c << gf.components.straight(length=10, width=bus_width)
straight_up2.connect('o1', bend_up.ports['o2'])

gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', straight_up2.ports['o2'])

# Lower output - SYMMETRIC (same lengths as upper)
straight_down1 = c << gf.components.straight(length=5, width=bus_width)
straight_down1.connect('o1', splitter.ports['o3'])

bend_down = c << gf.components.bend_circular(radius=bend_radius, angle=-90)
bend_down.connect('o1', straight_down1.ports['o2'])

straight_down2 = c << gf.components.straight(length=10, width=bus_width)
straight_down2.connect('o1', bend_down.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', straight_down2.ports['o2'])

# Add 2 more GCs in vertical array (total 5)
gc_spacing = 8.0
for i in range(2):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_out2.center[0], gc_out2.center[1] - 10 - i * gc_spacing))

