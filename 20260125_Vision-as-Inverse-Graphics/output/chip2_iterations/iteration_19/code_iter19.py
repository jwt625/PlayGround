import gdsfactory as gf

# Iteration 19: Start from iter9 (75%) and add ONLY the most critical missing feature
# VLM consistently mentioned: electrode routing traces to pads
# Keep everything else from iter9 that worked

c = gf.Component("chip2_iter19")

# Exact parameters from iteration 9 (which achieved 75%)
ring_radius = 10.0
ring_width = 0.5
coupling_gap = 0.20  # This was critical - better than 0.25
ring_spacing = 13.0
bus_width = 0.5

# Bus waveguide
bus_length = 70
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)

# Three rings
rings = []
for i in range(3):
    ring = c << gf.components.ring(radius=ring_radius, width=ring_width)
    x_pos = 12 + i * ring_spacing
    y_pos = bus_width / 2 + coupling_gap + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Electrodes - but this time add ROUTING to pads
electrode_radii = [11.0, 10.75, 10.5, 10.25, 10.0, 9.75, 9.5, 9.25, 9.0]
electrode_width = 0.3

for i, ring in enumerate(rings):
    # Add concentric electrodes
    for elec_radius in electrode_radii:
        elec = c << gf.components.ring(
            radius=elec_radius,
            width=electrode_width,
            layer=(2, 0)
        )
        elec.move(ring.center)
    
    # NEW: Add routing trace from outermost electrode to pad
    # Trace goes upward from top of electrode ring
    trace_start_x = ring.center[0]
    trace_start_y = ring.center[1] + electrode_radii[0] + electrode_width/2
    trace_length = 25

    # Vertical trace using rectangle on metal layer
    trace = c << gf.components.rectangle(size=(0.5, trace_length), layer=(2, 0))
    trace.move((trace_start_x - 0.25, trace_start_y))

    # Pad at top
    pad_size = 10
    pad = c << gf.components.rectangle(size=(pad_size, pad_size), layer=(2, 0))
    pad.move((trace_start_x - pad_size/2, trace_start_y + trace_length))

# Input GC
gc_input = c << gf.components.grating_coupler_elliptical_te()
gc_input.connect('o1', bus.ports['o1'])

# Splitter
splitter = c << gf.components.mmi1x2(width=bus_width, width_taper=1.0, length_taper=10, length_mmi=10)
splitter.connect('o1', bus.ports['o2'])

# Output routing (same as iter9)
bend_radius = 10.0

# Upper output
straight_up1 = c << gf.components.straight(length=5, width=bus_width)
straight_up1.connect('o1', splitter.ports['o2'])

bend_up = c << gf.components.bend_circular(radius=bend_radius, angle=90)
bend_up.connect('o1', straight_up1.ports['o2'])

straight_up2 = c << gf.components.straight(length=10, width=bus_width)
straight_up2.connect('o1', bend_up.ports['o2'])

gc_out1 = c << gf.components.grating_coupler_elliptical_te()
gc_out1.connect('o1', straight_up2.ports['o2'])

# Lower output
straight_down1 = c << gf.components.straight(length=5, width=bus_width)
straight_down1.connect('o1', splitter.ports['o3'])

bend_down = c << gf.components.bend_circular(radius=bend_radius, angle=-90)
bend_down.connect('o1', straight_down1.ports['o2'])

straight_down2 = c << gf.components.straight(length=10, width=bus_width)
straight_down2.connect('o1', bend_down.ports['o2'])

gc_out2 = c << gf.components.grating_coupler_elliptical_te()
gc_out2.connect('o1', straight_down2.ports['o2'])

# Additional GCs
gc_spacing = 8.0
for i in range(2):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc_out2.center[0], gc_out2.center[1] - 10 - i * gc_spacing))

