import gdsfactory as gf

# Create main component
c = gf.Component("chip2_iter05")

# Parameters from detailed analysis and topology clarification
ring_radius = 10.0  # µm
ring_width = 0.5    # µm
coupling_gap = 0.25 # µm (critical!)
ring_spacing = 15.0 # µm (center-to-center, adjusted)
bus_width = 0.5     # µm

# Create continuous bus waveguide running horizontally
bus_length = 70  # adjusted length
bus = c << gf.components.straight(length=bus_length, width=bus_width)
bus.movey(0)  # position at y=0

# Create three identical circular rings positioned to couple with bus
# Rings are side-coupled to the bus (parallel configuration)
rings = []
for i in range(3):
    # Create circular ring
    ring = c << gf.components.ring(
        radius=ring_radius,
        width=ring_width
    )
    
    # Position ring: bottom edge at coupling_gap from bus top edge
    x_pos = 12 + i * ring_spacing
    y_pos = coupling_gap + bus_width / 2 + ring_radius
    ring.move((x_pos, y_pos))
    rings.append(ring)

# Add metal electrodes OVER each ring (concentric structures)
# These are for electro-optic modulation
electrode_radius_outer = ring_radius + 2.0  # µm
electrode_radius_inner = ring_radius - 2.0  # µm
electrode_width = 1.0  # µm

for i, ring in enumerate(rings):
    # Outer electrode ring
    elec_outer = c << gf.components.ring(
        radius=electrode_radius_outer,
        width=electrode_width,
        layer=(2, 0)  # Metal layer
    )
    elec_outer.move(ring.center)
    
    # Inner electrode ring
    elec_inner = c << gf.components.ring(
        radius=electrode_radius_inner,
        width=electrode_width,
        layer=(2, 0)  # Metal layer
    )
    elec_inner.move(ring.center)

# Add bus splitter at the end
# Bus splits into 2 branches for dual output
split_x = bus.xmax + 2
bend_radius = 5.0

# Upper branch
bend_up = c << gf.components.bend_circular(radius=bend_radius, angle=90)
bend_up.connect('o1', bus.ports['o2'])

straight_up = c << gf.components.straight(length=10, width=bus_width)
straight_up.connect('o1', bend_up.ports['o2'])

# Lower branch (mirror of upper)
bend_down = c << gf.components.bend_circular(radius=bend_radius, angle=-90)
bend_down.connect('o1', bus.ports['o2'])

straight_down = c << gf.components.straight(length=10, width=bus_width)
straight_down.connect('o1', bend_down.ports['o2'])

# Add grating couplers at outputs
gc1 = c << gf.components.grating_coupler_elliptical_te()
gc1.connect('o1', straight_up.ports['o2'])

gc2 = c << gf.components.grating_coupler_elliptical_te()
gc2.connect('o1', straight_down.ports['o2'])

# Add 3 more grating couplers in vertical array (as seen in target)
gc_spacing = 8.0
for i in range(3):
    gc = c << gf.components.grating_coupler_elliptical_te()
    gc.move((gc1.center[0] + 10, gc1.center[1] - 10 - i * gc_spacing))

# Add input/output ports
c.add_port("input", port=bus.ports["o1"])

